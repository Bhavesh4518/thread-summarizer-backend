const express = require('express');
const NodeCache = require('node-cache');
const geminiService = require('../utils/gemini');
const router = express.Router();

// Cache setup
const cache = new NodeCache({ stdTTL: 1800 }); // 30 minutes

// Enhanced rate limiting
const { RateLimiterMemory } = require('rate-limiter-flexible');

const summaryRateLimiter = new RateLimiterMemory({
  points: 10, // 10 requests
  duration: 60, // per minute
});

const replyRateLimiter = new RateLimiterMemory({
  points: 5, // 5 requests
  duration: 60, // per minute
});

// Cache middleware
function cacheMiddleware(req, res, next) {
  const cacheKey = req.originalUrl + JSON.stringify(req.body);
  const cachedResponse = cache.get(cacheKey);
  
  if (cachedResponse) {
    return res.json({ ...cachedResponse, fromCache: true });
  }
  
  // Override res.json to cache the response
  const originalJson = res.json;
  res.json = function(body) {
    cache.set(cacheKey, body);
    return originalJson.call(this, { ...body, fromCache: false });
  };
  
  next();
}

// Rate limiting middleware
const rateLimitMiddleware = (rateLimiter) => async (req, res, next) => {
  try {
    const clientId = req.ip || req.headers['x-forwarded-for'] || 'anonymous';
    await rateLimiter.consume(clientId);
    next();
  } catch (rejRes) {
    res.status(429).json({ 
      error: 'Rate limit exceeded',
      message: 'Too many requests. Please try again in a minute.',
      retryAfter: 60
    });
  }
};

// Summarize thread endpoint
router.post('/summarize', cacheMiddleware, rateLimitMiddleware(summaryRateLimiter), async (req, res) => {
  try {
    const { threadContent } = req.body;
    
    if (!threadContent || !threadContent.text) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content is required' 
      });
    }
    
    const summary = await geminiService.summarizeThread(threadContent);
    res.json({ success: true, summary });
  } catch (error) {
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to summarize thread' 
    });
  }
});

// Generate reply endpoint
router.post('/reply', cacheMiddleware, rateLimitMiddleware(replyRateLimiter), async (req, res) => {
  try {
    const { threadContent, summary } = req.body;
    
    if (!threadContent || !summary) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content and summary are required' 
      });
    }
    
    if (!threadContent.text) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content text is required' 
      });
    }
    
    if (!Array.isArray(summary.keyPoints)) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Summary must include keyPoints array' 
      });
    }
    
    const reply = await geminiService.generateReply(threadContent, summary);
    res.json({ success: true, reply });
  } catch (error) {
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to generate reply' 
    });
  }
});

// Cache management endpoint (for testing)
router.post('/clear-cache', (req, res) => {
  cache.flushAll();
  res.json({ success: true, message: 'Cache cleared' });
});

module.exports = router;
