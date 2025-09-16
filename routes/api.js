const express = require('express');
const geminiService = require('../utils/gemini'); // Changed from openai
const router = express.Router();

// Rate limiting middleware (simple version)
const rateLimit = {};
const RATE_LIMIT_COUNT = 10; // 10 requests per hour
const RATE_LIMIT_WINDOW = 60 * 60 * 1000; // 1 hour

router.use((req, res, next) => {
  const clientId = req.headers['x-client-id'] || 'anonymous';
  const now = Date.now();
  
  if (!rateLimit[clientId]) {
    rateLimit[clientId] = { count: 0, resetTime: now + RATE_LIMIT_WINDOW };
  }
  
  if (now > rateLimit[clientId].resetTime) {
    rateLimit[clientId] = { count: 0, resetTime: now + RATE_LIMIT_WINDOW };
  }
  
  if (rateLimit[clientId].count >= RATE_LIMIT_COUNT) {
    return res.status(429).json({ 
      error: 'Rate limit exceeded. Please try again later.' 
    });
  }
  
  rateLimit[clientId].count++;
  next();
});

// Summarize thread endpoint
router.post('/summarize', async (req, res) => {
  try {
    const { threadContent } = req.body;
    
    if (!threadContent || !threadContent.text) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content is required' 
      });
    }
    
    if (threadContent.text.length > 10000) {
      return res.status(400).json({
        error: 'Bad Request',
        message: 'Thread content too long. Please provide shorter content.'
      });
    }
    
    const summary = await geminiService.summarizeThread(threadContent); // Changed
    res.json({ success: true, summary });
  } catch (error) {
    console.error('Summarization error:', error);
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to summarize thread' 
    });
  }
});

// Generate reply endpoint
router.post('/reply', async (req, res) => {
  try {
    const { threadContent, summary } = req.body;
    
    if (!threadContent || !summary) {
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content and summary are required' 
      });
    }
    
    const reply = await geminiService.generateReply(threadContent, summary); // Changed
    res.json({ success: true, reply });
  } catch (error) {
    console.error('Reply generation error:', error);
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to generate reply' 
    });
  }
});

module.exports = router;