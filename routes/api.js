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
  console.log('📥 Received summarize request');
  console.log('📄 Content length:', req.body.threadContent?.text?.length || 0);
  
  try {
    const { threadContent } = req.body;
    
    if (!threadContent || !threadContent.text) {
      console.log('❌ Missing thread content');
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content is required' 
      });
    }
    
    console.log('🤖 Calling Gemini API...');
    const summary = await geminiService.summarizeThread(threadContent);
    console.log('✅ Gemini summary generated');
    
    res.json({ success: true, summary });
  } catch (error) {
    console.error('💥 Summarization error:', error);
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to summarize thread' 
    });
  }
});

// Generate reply endpoint
router.post('/reply', async (req, res) => {
  console.log('📥 Received reply request');
  
  try {
    const { threadContent, summary } = req.body;
    
    if (!threadContent || !summary) {
      console.log('❌ Missing content or summary');
      return res.status(400).json({ 
        error: 'Bad Request',
        message: 'Thread content and summary are required' 
      });
    }
    
    console.log('🤖 Calling Gemini API for reply...');
    const reply = await geminiService.generateReply(threadContent, summary);
    console.log('✅ Gemini reply generated');
    
    res.json({ success: true, reply });
  } catch (error) {
    console.error('💥 Reply generation error:', error);
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: error.message || 'Failed to generate reply' 
    });
  }
});

module.exports = router;
