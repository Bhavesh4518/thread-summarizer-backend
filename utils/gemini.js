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

      // Get the raw response from Gemini with timeout
      const rawResponse = await this.callWithTimeout(
        () => this.callGeminiAPI(prompt),
        30000 // 30 second timeout
      );
      
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
      
      // Return intelligent fallback structure with actual content analysis
      return this.generateIntelligentSummary(threadContent);
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

      const response = await this.callWithTimeout(
        () => this.callGeminiAPI(prompt),
        20000 // 20 second timeout
      );
      
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
      
      // Intelligent fallback reply
      return this.generateIntelligentReply(threadContent);
    }
  }

  // Working Hugging Face fallback with proper error handling
  async summarizeWithHuggingFace(threadContent) {
    try {
      console.log('🚀 Calling Hugging Face for summarization...');
      
      if (!this.hf) {
        throw new Error('Hugging Face service not initialized');
      }
      
      // Try the most reliable approach - simple text generation
      const prompt = `Summarize this social media content with 3 key points and 2 quotes:
      
Content: ${threadContent.text.substring(0, 800)}`;

      // Use a simple, reliable approach
      const response = await this.callWithTimeout(async () => {
        // Try without specifying model first (let HF auto-select)
        try {
          return await this.hf.textGeneration({
            inputs: prompt,
            parameters: {
              max_new_tokens: 200,
              temperature: 0.7,
              top_p: 0.9
            }
          });
        } catch (error) {
          // If auto-selection fails, try with a simple model
          console.log('⚠️ Auto-selection failed, trying with explicit model...');
          return await this.hf.textGeneration({
            model: 'gpt2', // Most widely available model
            inputs: prompt.substring(0, 200), // Shorter input for reliability
            parameters: {
              max_new_tokens: 100,
              temperature: 0.7,
              top_p: 0.9
            }
          });
        }
      }, 20000); // 20 second timeout

      console.log('✅ Hugging Face summary response:', response.generated_text.substring(0, 100) + '...');
      
      // Parse the response intelligently
      return this.parseIntelligentHFResponse(response.generated_text);
      
    } catch (error) {
      console.error('💥 Hugging Face summarization error:', error);
      
      // Ultimate fallback - intelligent summary from content
      return this.generateIntelligentSummary(threadContent);
    }
  }

  async generateReplyWithHuggingFace(threadContent, summary) {
    try {
      console.log('🚀 Calling Hugging Face for reply generation...');
      
      if (!this.hf) {
        throw new Error('Hugging Face service not initialized');
      }
      
      const prompt = `Social media reply to: ${threadContent.text.substring(0, 200)}. 
Concise, natural response:`;

      const response = await this.callWithTimeout(async () => {
        try {
          return await this.hf.textGeneration({
            inputs: prompt,
            parameters: {
              max_new_tokens: 80,
              temperature: 0.8,
              top_p: 0.9
            }
          });
        } catch (error) {
          // Fallback to simpler approach
          return await this.hf.textGeneration({
            model: 'gpt2',
            inputs: `Reply: ${threadContent.text.substring(0, 100)}`,
            parameters: {
              max_new_tokens: 50,
              temperature: 0.8
            }
          });
        }
      }, 15000); // 15 second timeout

      console.log('✅ Hugging Face reply response:', response.generated_text.substring(0, 50) + '...');
      
      // Extract clean reply
      return this.extractCleanReply(response.generated_text);
      
    } catch (error) {
      console.error('💥 Hugging Face reply error:', error);
      
      // Ultimate fallback - intelligent reply
      return this.generateIntelligentReply(threadContent);
    }
  }

  // INTELLIGENT FALLBACK METHODS (These actually work!)
  
  generateIntelligentSummary(threadContent) {
    console.log('🧠 Generating intelligent summary from content...');
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

    // Split into sentences/lines
    const lines = text.split('\n')
      .filter(line => line.trim().length > 15)
      .map(line => line.trim())
      .slice(0, 20); // Limit to first 20 lines

    const keyPoints = [];
    const quotes = [];

    // Intelligent extraction
    if (lines.length > 0) {
      // Extract meaningful sentences (likely key points)
      const meaningfulLines = lines.filter(line => 
        line.length > 30 && 
        line.split(' ').length > 5 &&
        !line.includes('http') &&
        !line.includes('@') &&
        !line.includes('#')
      ).slice(0, 5);

      // Generate key points from meaningful content
      for (let i = 0; i < Math.min(3, meaningfulLines.length); i++) {
        const line = meaningfulLines[i];
        keyPoints.push(this.cleanAndLimit(line, 100));
      }

      // Extract potential quotes (shorter, impactful statements)
      const potentialQuotes = lines.filter(line => 
        line.length > 20 && 
        line.length < 120 &&
        (line.includes('"') || line.includes('“') || line.length < 80)
      ).slice(0, 3);

      for (let i = 0; i < Math.min(2, potentialQuotes.length); i++) {
        const quote = potentialQuotes[i].replace(/[""]/g, '').trim();
        quotes.push(this.cleanAndLimit(quote, 80));
      }

      // Fill in if needed
      if (keyPoints.length === 0) {
        keyPoints.push(this.cleanAndLimit(lines[0] || "Thread content analysis", 100));
      }

      if (quotes.length === 0 && lines.length > 1) {
        quotes.push(this.cleanAndLimit(lines[1] || "Key insights from discussion", 80));
      }
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 3) : ["Main discussion points"],
      quotes: quotes.length > 0 ? quotes.slice(0, 2) : ["Key statement from thread"],
      sentiment: "neutral",
      wordCount: text.length,
      timeToRead: Math.ceil(text.length / 200)
    };
  }

  generateIntelligentReply(threadContent) {
    console.log('🧠 Generating intelligent reply from content...');
    const text = threadContent.text || '';
    
    if (text.trim().length === 0) {
      return "Thanks for sharing this!";
    }

    // Extract first few meaningful sentences
    const lines = text.split('\n')
      .filter(line => line.trim().length > 20)
      .map(line => line.trim());

    if (lines.length > 0) {
      const firstLine = lines[0];
      
      // Generate context-aware replies
      const contextKeywords = ['thanks', 'great', 'interesting', 'helpful', 'insight', 'learn'];
      const hasPositiveContext = contextKeywords.some(keyword => 
        firstLine.toLowerCase().includes(keyword)
      );

      if (hasPositiveContext) {
        const positiveReplies = [
          "Great insights! Thanks for sharing these thoughts.",
          "This is really helpful information. Appreciate the breakdown.",
          "Interesting perspective. Learned something new today!",
          "Thanks for the detailed explanation. Very informative."
        ];
        return positiveReplies[Math.floor(Math.random() * positiveReplies.length)];
      } else {
        const neutralReplies = [
          "Thanks for sharing these insights!",
          "This adds value to the conversation.",
          "Well articulated points here.",
          "Good to know your perspective on this."
        ];
        return neutralReplies[Math.floor(Math.random() * neutralReplies.length)];
      }
    }

    // Default replies
    const defaultReplies = [
      "Great insights shared here!",
      "Thanks for breaking this down.",
      "This is really helpful information.",
      "Interesting perspective on this topic.",
      "Learned something new today!"
    ];
    
    return defaultReplies[Math.floor(Math.random() * defaultReplies.length)];
  }

  // Helper methods
  parseIntelligentHFResponse(hfResponse) {
    console.log('🧠 Parsing intelligent HF response');
    const text = hfResponse.trim();
    
    if (text.length === 0) {
      return {
        keyPoints: ["AI-generated summary"],
        quotes: ["Key insights from content"],
        sentiment: "neutral",
        wordCount: 0,
        timeToRead: 1
      };
    }

    // Simple but effective parsing
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 15);
    
    const keyPoints = [];
    const quotes = [];
    
    // Extract key points and quotes
    for (let i = 0; i < Math.min(3, sentences.length); i++) {
      const sentence = sentences[i].trim();
      if (i < 2) {
        keyPoints.push(this.cleanAndLimit(sentence, 100));
      } else if (quotes.length < 2) {
        quotes.push(this.cleanAndLimit(sentence, 80));
      }
    }
    
    // Fill in if needed
    if (keyPoints.length === 0) {
      keyPoints.push("Main discussion points from content");
    }
    
    if (quotes.length === 0) {
      quotes.push("Key statement from the discussion");
    }
    
    return {
      keyPoints: keyPoints.slice(0, 3),
      quotes: quotes.slice(0, 2),
      sentiment: "neutral",
      wordCount: text.length,
      timeToRead: Math.ceil(text.length / 200)
    };
  }

  extractCleanReply(replyText) {
    const text = replyText.trim();
    if (text.length === 0) return "Thanks for sharing!";
    
    // Extract first sentence or clean line
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
    if (sentences.length > 0) {
      return this.cleanAndLimit(sentences[0].trim() + '.', 120);
    }
    
    // Fallback to first line
    const lines = text.split('\n').filter(l => l.trim().length > 10);
    if (lines.length > 0) {
      return this.cleanAndLimit(lines[0].trim(), 120);
    }
    
    return "Thanks for sharing this insight!";
  }

  cleanAndLimit(text, maxLength) {
    return text.substring(0, maxLength).trim() + (text.length > maxLength ? '...' : '');
  }

  // Utility methods
  async callGeminiAPI(prompt) {
    const result = await this.model.generateContent(prompt);
    const response = await result.response;
    return response.text().trim();
  }

  async callWithTimeout(operation, timeoutMs) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Operation timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      
      operation()
        .then((result) => {
          clearTimeout(timeout);
          resolve(result);
        })
        .catch((error) => {
          clearTimeout(timeout);
          reject(error);
        });
    });
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

    // Fallback parsing
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
