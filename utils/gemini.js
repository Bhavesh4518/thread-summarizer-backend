const { GoogleGenerativeAI } = require("@google/generative-ai");
const { HfInference } = require('@huggingface/inference');

class GeminiService {
  constructor() {
    this.genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    this.model = this.genAI.getGenerativeModel({ 
      model: process.env.GEMINI_MODEL || "gemini-1.5-flash" 
    });
    
    // Initialize Hugging Face service if API key is available
    this.HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
    if (this.HF_TOKEN) {
      this.hf = new HfInference(this.HF_TOKEN);
      console.log('🚀 Hugging Face service initialized');
    } else {
      console.warn('⚠️ Hugging Face API key not found - fallback will not work');
    }
  }

  async summarizeThread(threadContent) {
    try {
      console.log('🚀 Calling Gemini API for summarization...');
      
      const prompt = `You are a content summarization expert. Create a concise, actionable summary of this social media thread.

Rules:
- Provide EXACTLY 3 key points (numbered 1, 2, 3)
- Include EXACTLY 2 notable quotes (labeled as "Quote 1:" and "Quote 2:")
- Keep each key point under 100 characters
- Keep each quote under 80 characters
- Focus on the main discussion, ignore replies/suggestions
- Use clear, simple language
- Format exactly as shown below:

Key Points:
1. [First key point here]
2. [Second key point here]
3. [Third key point here]

Quotes:
Quote 1: [First notable quote here]
Quote 2: [Second notable quote here]

Thread content: ${threadContent.text.substring(0, 2500)}`;

      // Get the raw response from Gemini
      const rawResponse = await this.retryOperation(() => this.callGeminiAPI(prompt));
      
      // Parse and structure the response
      const parsedResponse = this.parseSummaryResponse(rawResponse);
      
      console.log('🤖 Gemini summary response:', {
        raw: rawResponse.substring(0, 100) + '...',
        parsed: parsedResponse
      });
      
      return parsedResponse;
    } catch (error) {
      console.error('💥 Gemini summary error:', error.message);
      
      // Check if it's a quota error and fallback to Hugging Face
      if ((error.message.includes('429') || error.message.includes('quota')) && this.hf) {
        console.log('🔄 Gemini quota exceeded, falling back to Hugging Face...');
        try {
          return await this.summarizeWithHuggingFace(threadContent);
        } catch (hfError) {
          console.error('💥 Hugging Face fallback also failed:', hfError.message);
        }
      }
      
      // Return fallback structure
      return {
        keyPoints: ["Content summary will appear here"],
        quotes: ["Notable quotes will appear here"],
        sentiment: "neutral",
        wordCount: 0,
        timeToRead: 1
      };
    }
  }

  async generateReply(threadContent, summary) {
    try {
      console.log('🚀 Calling Gemini API for reply generation...');
      
      const prompt = `Generate a human-like response to this thread that:
- Sounds natural and conversational
- Adds value to the discussion
- Matches the tone of the original content
- Avoids AI-detection patterns
- Is 1-2 sentences maximum (under 120 characters total)

Thread content: ${threadContent.text.substring(0, 1500)}

Summary key points: ${summary.keyPoints.slice(0, 2).join(', ')}

Generate only the response text, nothing else. Keep it concise and natural.`;

      const response = await this.retryOperation(() => this.callGeminiAPI(prompt));
      console.log('🤖 Gemini reply response:', response.substring(0, 50) + '...');
      return response;
    } catch (error) {
      console.error('💥 Gemini reply error:', error.message);
      
      // Check if it's a quota error and fallback to Hugging Face
      if ((error.message.includes('429') || error.message.includes('quota')) && this.hf) {
        console.log('🔄 Gemini quota exceeded, falling back to Hugging Face...');
        try {
          return await this.generateReplyWithHuggingFace(threadContent, summary);
        } catch (hfError) {
          console.error('💥 Hugging Face fallback also failed:', hfError.message);
        }
      }
      
      return "Thanks for sharing these insights!";
    }
  }

  // Hugging Face fallback methods with proper task handling
  async summarizeWithHuggingFace(threadContent) {
    try {
      console.log('🚀 Calling Hugging Face for summarization...');
      
      if (!this.hf) {
        throw new Error('Hugging Face service not initialized');
      }
      
      // Try summarization models first (proper task)
      const summarizationModels = [
        'facebook/bart-large-cnn',
        'sshleifer/distilbart-cnn-12-6'
      ];
      
      // Try summarization models
      for (const model of summarizationModels) {
        try {
          console.log(`🔍 Trying summarization model: ${model}`);
          
          const response = await this.hf.summarization({
            model: model,
            inputs: threadContent.text.substring(0, 1000),
            parameters: {
              max_length: 150,
              min_length: 50,
              do_sample: false
            }
          });
          
          console.log(`✅ Hugging Face summarization with ${model}:`, response.summary_text.substring(0, 100) + '...');
          
          // Convert summary to our format
          return this.convertHFSummaryToOurFormat(response.summary_text);
        } catch (modelError) {
          console.warn(`⚠️ Summarization model ${model} failed:`, modelError.message);
          continue;
        }
      }
      
      // Fallback to text generation models
      const generationModels = [
        'google/flan-t5-base',
        'gpt2'
      ];
      
      for (const model of generationModels) {
        try {
          console.log(`🔍 Trying text generation model: ${model}`);
          
          const prompt = `Summarize this social media thread with exactly 3 key points and 2 notable quotes:
          
Thread content: ${threadContent.text.substring(0, 800)}

Format your response as:
Key Points:
1. [First key point]
2. [Second key point]  
3. [Third key point]

Quotes:
Quote 1: [First notable quote]
Quote 2: [Second notable quote]`;

          const response = await this.hf.textGeneration({
            model: model,
            inputs: prompt,
            parameters: {
              max_new_tokens: 200,
              temperature: 0.7,
              top_p: 0.9
            }
          });

          console.log(`✅ Hugging Face text generation with ${model}:`, response.generated_text.substring(0, 100) + '...');
          
          return this.parseSummaryResponse(response.generated_text);
        } catch (modelError) {
          console.warn(`⚠️ Text generation model ${model} failed:`, modelError.message);
          continue;
        }
      }
      
      throw new Error('All Hugging Face models failed');
      
    } catch (error) {
      console.error('💥 Hugging Face summarization error:', error);
      
      // Ultimate fallback - basic summary
      return this.generateBasicSummary(threadContent);
    }
  }

  async generateReplyWithHuggingFace(threadContent, summary) {
    try {
      console.log('🚀 Calling Hugging Face for reply generation...');
      
      if (!this.hf) {
        throw new Error('Hugging Face service not initialized');
      }
      
      // Try text generation models for replies
      const modelsToTry = [
        'google/flan-t5-base',
        'microsoft/DialoGPT-medium',
        'gpt2'
      ];
      
      for (const model of modelsToTry) {
        try {
          console.log(`🔍 Trying reply generation with model: ${model}`);
          
          const prompt = `Generate a human-like response to this social media thread:
          
Thread content: ${threadContent.text.substring(0, 500)}
Summary key points: ${summary.keyPoints.slice(0, 2).join(', ')}

Generate only a concise, natural response (1-2 sentences):`;

          const response = await this.hf.textGeneration({
            model: model,
            inputs: prompt,
            parameters: {
              max_new_tokens: 100,
              temperature: 0.8,
              top_p: 0.9
            }
          });

          console.log(`✅ Hugging Face reply with ${model}:`, response.generated_text.substring(0, 50) + '...');
          
          return response.generated_text.trim();
        } catch (modelError) {
          console.warn(`⚠️ Reply model ${model} failed:`, modelError.message);
          continue;
        }
      }
      
      throw new Error('All Hugging Face reply models failed');
      
    } catch (error) {
      console.error('💥 Hugging Face reply error:', error);
      
      // Ultimate fallback - basic reply
      return this.generateBasicReply();
    }
  }

  // Convert Hugging Face summary to our format
  convertHFSummaryToOurFormat(hfSummary) {
    console.log('🔄 Converting HF summary to our format');
    
    // Simple conversion - split into key points and quotes
    const sentences = hfSummary.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    const keyPoints = [];
    const quotes = [];
    
    // Extract key points from sentences
    for (let i = 0; i < Math.min(3, sentences.length); i++) {
      if (i < 2) {
        keyPoints.push(sentences[i].trim().substring(0, 100));
      } else if (quotes.length < 2) {
        quotes.push(sentences[i].trim().substring(0, 80));
      }
    }
    
    // Fill in if needed
    if (keyPoints.length === 0) {
      keyPoints.push("Main discussion points from the thread");
    }
    
    if (quotes.length === 0) {
      quotes.push("Notable statement from the discussion");
    }
    
    return {
      keyPoints: keyPoints,
      quotes: quotes,
      sentiment: "neutral",
      wordCount: hfSummary.length,
      timeToRead: Math.ceil(hfSummary.length / 200)
    };
  }

  // Basic fallback methods
  generateBasicSummary(threadContent) {
    console.log('⚠️ Using basic summary fallback');
    const text = threadContent.text || '';
    
    if (text.trim().length === 0) {
      return {
        keyPoints: ["No content found to summarize"],
        quotes: ["No quotes available"],
        sentiment: "neutral",
        wordCount: 0,
        timeToRead: 0
      };
    }

    const lines = text.split('\n')
      .filter(line => line.trim().length > 10)
      .map(line => line.trim());

    const keyPoints = [];
    const quotes = [];

    const meaningfulLines = lines.filter(line => 
      line.length > 30 && 
      line.split(' ').length > 5 &&
      !line.includes('http')
    );

    for (let i = 0; i < Math.min(3, meaningfulLines.length); i++) {
      const line = meaningfulLines[i];
      keyPoints.push(line.substring(0, 100) + (line.length > 100 ? '...' : ''));
    }

    if (keyPoints.length === 0 && lines.length > 0) {
      keyPoints.push(lines[0].substring(0, 100) + (lines[0].length > 100 ? '...' : ''));
    }

    const shortLines = lines.filter(line => 
      line.length > 20 && line.length < 100
    );

    for (let i = 0; i < Math.min(2, shortLines.length); i++) {
      quotes.push(shortLines[i]);
    }

    if (quotes.length === 0 && lines.length > 1) {
      quotes.push(lines[Math.floor(lines.length / 2)].substring(0, 80) + '...');
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints : ["Thread content analysis"],
      quotes: quotes.length > 0 ? quotes : ["Key insights from discussion"],
      sentiment: "neutral",
      wordCount: text.length,
      timeToRead: Math.ceil(text.length / 200)
    };
  }

  generateBasicReply() {
    console.log('⚠️ Using basic reply fallback');
    const sampleReplies = [
      "Great insights shared here! Thanks for breaking this down.",
      "This is really helpful information. Appreciate the detailed explanation.",
      "Interesting perspective on this topic. Learned something new today!",
      "Thanks for sharing these thoughts. Very informative thread.",
      "This adds a lot of value to the conversation. Well articulated!"
    ];
    
    return sampleReplies[Math.floor(Math.random() * sampleReplies.length)];
  }

  async callGeminiAPI(prompt) {
    const result = await this.model.generateContent(prompt);
    const response = await result.response;
    return response.text().trim();
  }

  async retryOperation(operation, maxRetries = 3, delay = 1000) {
    let lastError;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        // If it's not a 503 error, don't retry
        if (!error.message.includes('503') && !error.message.includes('overloaded')) {
          // Check for quota errors to trigger fallback immediately
          if (error.message.includes('429') || error.message.includes('quota')) {
            throw error;
          }
          throw error;
        }
        
        // If this is the last retry, throw the error
        if (i === maxRetries - 1) {
          throw error;
        }
        
        // Wait before retrying (exponential backoff)
        const waitTime = delay * Math.pow(2, i);
        console.log(`⏳ Retrying in ${waitTime}ms... (attempt ${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
    
    throw lastError;
  }

  parseSummaryResponse(aiResponse) {
    console.log('🔍 Parsing AI response:', aiResponse.substring(0, 100) + '...');
    
    const keyPoints = [];
    const quotes = [];
    let sentiment = "neutral";
    let timeToRead = 1;

    // Split into lines and clean
    const lines = aiResponse.split('\n').filter(line => line.trim());

    // Parse structured format
    let currentSection = '';
    lines.forEach(line => {
      const cleanLine = line.trim();
      const lowerLine = cleanLine.toLowerCase();
      
      // Detect sections
      if (lowerLine.includes('key points') || lowerLine.includes('main points')) {
        currentSection = 'points';
      } else if (lowerLine.includes('quotes') || lowerLine.includes('notable quotes')) {
        currentSection = 'quotes';
      } else if (lowerLine.includes('sentiment')) {
        sentiment = lowerLine.includes('positive') ? 'positive' : 
                   lowerLine.includes('negative') ? 'negative' : 'neutral';
      } else if (lowerLine.includes('reading time')) {
        const timeMatch = line.match(/(\d+)/);
        if (timeMatch) timeToRead = parseInt(timeMatch[1]);
      } else if (line.trim().startsWith('-') || line.trim().startsWith('*') || line.trim().match(/^\d+\./)) {
        const cleanLine = line.replace(/^[-*\d.]+\s*/, '').trim();
        if (currentSection === 'points' && cleanLine) {
          if (keyPoints.length < 3) {
            keyPoints.push(cleanLine.substring(0, 100));
          }
        } else if (currentSection === 'quotes' && cleanLine) {
          if (quotes.length < 2) {
            quotes.push(cleanLine.replace(/[""]/g, '').substring(0, 80));
          }
        }
      }
    });

    // Fallback parsing if structured parsing didn't work
    if (keyPoints.length === 0 && quotes.length === 0) {
      const cleanLines = lines.filter(line => line.length > 20 && line.length < 150);
      for (let i = 0; i < Math.min(3, cleanLines.length); i++) {
        if (i < 2) {
          keyPoints.push(cleanLines[i].substring(0, 100));
        } else if (quotes.length < 2) {
          quotes.push(cleanLines[i].substring(0, 80));
        }
      }
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 3) : ["Main discussion points"],
      quotes: quotes.length > 0 ? quotes.slice(0, 2) : ["Key statement from thread"],
      sentiment: sentiment,
      wordCount: 0,
      timeToRead: timeToRead || Math.max(1, Math.floor(lines.length / 50))
    };
  }
}

module.exports = new GeminiService();
