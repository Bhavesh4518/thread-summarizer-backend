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
      
      // Return intelligent fallback structure
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
      
      return this.generateIntelligentReply(threadContent);
    }
  }

  // Hugging Face fallback with your recommended models
  async summarizeWithHuggingFace(threadContent) {
    try {
      console.log('🚀 Calling Hugging Face for summarization...');
      
      if (!this.hf) {
        throw new Error('Hugging Face service not initialized');
      }
      
      // Try your recommended models in order of preference
      const modelsToTry = [
        'sshleifer/distilbart-cnn-12-6', // Lighter, faster version
        'facebook/bart-large-cnn',       // High-quality abstractive
        'google/t5-small',               // Versatile, multi-task
        'google/t5-base'                 // Better speed/performance trade-off
      ];
      
      for (const model of modelsToTry) {
        try {
          console.log(`🔍 Trying Hugging Face model: ${model}`);
          
          // Use summarization task specifically for these models
          let response;
          
          if (model.includes('t5')) {
            // For T5 models, use text-to-text format
            response = await this.callWithTimeout(async () => {
              return await this.hf.textGeneration({
                model: model,
                inputs: `summarize: ${threadContent.text.substring(0, 800)}`,
                parameters: {
                  max_new_tokens: 200,
                  temperature: 0.7,
                  top_p: 0.9,
                  repetition_penalty: 1.2
                }
              });
            }, 25000); // 25 second timeout
          } else {
            // For BART models, use summarization task
            response = await this.callWithTimeout(async () => {
              return await this.hf.summarization({
                model: model,
                inputs: threadContent.text.substring(0, 1000),
                parameters: {
                  max_length: 150,
                  min_length: 50,
                  do_sample: false
                }
              });
            }, 30000); // 30 second timeout
          }
          
          console.log(`✅ Hugging Face summary with ${model}:`, response.generated_text ? response.generated_text.substring(0, 100) : response.summary_text.substring(0, 100) + '...');
          
          // Convert to our format
          const summaryText = response.generated_text || response.summary_text || '';
          return this.convertHFSummaryToOurFormat(summaryText);
          
        } catch (modelError) {
          console.warn(`⚠️ Model ${model} failed:`, modelError.message);
          continue;
        }
      }
      
      throw new Error('All recommended Hugging Face models failed');
      
    } catch (error) {
      console.error('💥 Hugging Face summarization error:', error);
      
      // Ultimate fallback - intelligent summary
      return this.generateIntelligentSummary(threadContent);
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
        'google/t5-base',      // Good for text generation
        'google/t5-small',     // Faster alternative
        'gpt2'                 // Always available fallback
      ];
      
      for (const model of modelsToTry) {
        try {
          console.log(`🔍 Trying reply generation with model: ${model}`);
          
          const prompt = `Generate a human-like response to this social media thread:
          
Thread content: ${threadContent.text.substring(0, 500)}
Summary key points: ${summary.keyPoints.slice(0, 2).join(', ')}

Generate only a concise, natural response (1-2 sentences):`;

          const response = await this.callWithTimeout(async () => {
            return await this.hf.textGeneration({
              model: model,
              inputs: prompt,
              parameters: {
                max_new_tokens: 100,
                temperature: 0.8,
                top_p: 0.9,
                repetition_penalty: 1.2
              }
            });
          }, 20000); // 20 second timeout

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
      
      // Ultimate fallback - intelligent reply
      return this.generateIntelligentReply(threadContent);
    }
  }

  // Convert Hugging Face summary to our format
  convertHFSummaryToOurFormat(hfSummary) {
    console.log('🔄 Converting HF summary to our format');
    
    if (!hfSummary || hfSummary.trim().length === 0) {
      return {
        keyPoints: ["AI-generated summary"],
        quotes: ["Key insights from content"],
        sentiment: "neutral",
        wordCount: 0,
        timeToRead: 1
      };
    }
    
    const text = hfSummary.trim();
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    const keyPoints = [];
    const quotes = [];
    
    // Extract key points and quotes from sentences
    for (let i = 0; i < Math.min(3, sentences.length); i++) {
      const sentence = sentences[i].trim();
      if (i < 2) {
        keyPoints.push(sentence.substring(0, 100) + (sentence.length > 100 ? '...' : ''));
      } else if (quotes.length < 2) {
        quotes.push(sentence.substring(0, 80) + (sentence.length > 80 ? '...' : ''));
      }
    }
    
    // Fill in if needed
    if (keyPoints.length === 0) {
      keyPoints.push("Main discussion points from the thread");
    }
    
    if (quotes.length === 0 && sentences.length > 0) {
      quotes.push(sentences[0].substring(0, 80) + '...');
    }
    
    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 3) : ["Main discussion points"],
      quotes: quotes.length > 0 ? quotes.slice(0, 2) : ["Key statement from thread"],
      sentiment: "neutral",
      wordCount: text.length,
      timeToRead: Math.ceil(text.length / 200)
    };
  }

  // Intelligent fallback methods (when all APIs fail)
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

    // Split into lines and clean
    const lines = text.split('\n')
      .filter(line => line.trim().length > 10)
      .map(line => line.trim())
      .slice(0, 30); // Limit for performance

    const keyPoints = [];
    const quotes = [];

    // Intelligent extraction
    if (lines.length > 0) {
      // Extract meaningful content (likely key points)
      const meaningfulLines = lines.filter(line => 
        line.length > 30 && 
        line.split(' ').length > 5 &&
        !line.includes('http') &&
        !line.includes('RT @') &&
        !line.startsWith('Replying to') &&
        line.length < 280
      );

      // Generate key points
      for (let i = 0; i < Math.min(3, meaningfulLines.length); i++) {
        const line = meaningfulLines[i];
        const cleanLine = line
          .replace(/@\w+/g, '')
          .replace(/#\w+/g, '')
          .replace(/\s+/g, ' ')
          .trim();
        
        if (cleanLine.length > 15) {
          keyPoints.push(cleanLine.substring(0, 100) + (cleanLine.length > 100 ? '...' : ''));
        }
      }

      // Extract potential quotes
      const potentialQuotes = lines.filter(line => 
        line.length > 20 && 
        line.length < 120 &&
        (line.includes('"') || line.includes('“') || line.length < 80)
      );

      for (let i = 0; i < Math.min(2, potentialQuotes.length); i++) {
        const quote = potentialQuotes[i]
          .replace(/[""]/g, '')
          .replace(/@\w+/g, '')
          .trim();
        quotes.push(quote.substring(0, 80) + (quote.length > 80 ? '...' : ''));
      }

      // Fill in if needed
      if (keyPoints.length === 0 && lines.length > 0) {
        const firstLine = lines[0]
          .replace(/@\w+/g, '')
          .replace(/#\w+/g, '')
          .replace(/\s+/g, ' ')
          .trim();
        keyPoints.push(firstLine.substring(0, 100) + (firstLine.length > 100 ? '...' : ''));
      }

      if (quotes.length === 0 && lines.length > 1) {
        const middleLine = lines[Math.floor(lines.length / 2)]
          .replace(/[""]/g, '')
          .replace(/@\w+/g, '')
          .trim();
        quotes.push(middleLine.substring(0, 80) + (middleLine.length > 80 ? '...' : ''));
      }
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 3) : ["Main discussion points from thread"],
      quotes: quotes.length > 0 ? quotes.slice(0, 2) : ["Key statement from discussion"],
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
      const positiveKeywords = ['thanks', 'great', 'amazing', 'awesome', 'love', 'excellent', 'fantastic'];
      const questionKeywords = ['?', 'how', 'what', 'why', 'when', 'where', 'can you'];
      const hasPositiveContext = positiveKeywords.some(keyword => 
        firstLine.toLowerCase().includes(keyword)
      );
      const hasQuestion = questionKeywords.some(keyword => 
        firstLine.toLowerCase().includes(keyword)
      );

      if (hasPositiveContext) {
        const positiveReplies = [
          "Great insights shared here! Thanks for breaking this down.",
          "This is really helpful information. Appreciate the detailed explanation.",
          "Interesting perspective on this topic. Learned something new today!",
          "Thanks for sharing these thoughts. Very informative thread.",
          "This adds a lot of value to the conversation. Well articulated!"
        ];
        return positiveReplies[Math.floor(Math.random() * positiveReplies.length)];
      } else if (hasQuestion) {
        const questionReplies = [
          "Good question! Here's my take on this...",
          "That's an interesting point. From my experience...",
          "I've been thinking about this too. Here's what I found...",
          "Thanks for raising this question. My perspective is..."
        ];
        return questionReplies[Math.floor(Math.random() * questionReplies.length)];
      } else {
        const neutralReplies = [
          "Thanks for sharing these insights!",
          "This is really helpful information.",
          "Interesting perspective, learned something new!",
          "Appreciate the detailed explanation.",
          "This adds value to the conversation."
        ];
        return neutralReplies[Math.floor(Math.random() * neutralReplies.length)];
      }
    }

    // Default replies
    const defaultReplies = [
      "Great insights shared here!",
      "This is really helpful information.",
      "Interesting perspective on this topic.",
      "Thanks for sharing these thoughts.",
      "This adds value to the conversation."
    ];
    
    return defaultReplies[Math.floor(Math.random() * defaultReplies.length)];
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
