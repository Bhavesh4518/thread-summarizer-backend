const { GoogleGenerativeAI } = require("@google/generative-ai");

class GeminiService {
  constructor() {
    this.genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    this.model = this.genAI.getGenerativeModel({ 
      model: process.env.GEMINI_MODEL || "gemini-1.5-flash" 
    });
  }

  async summarizeThread(threadContent) {
    try {
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
      console.error('💥 Gemini summary error:', error);
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
      console.error('💥 Gemini reply error:', error);
      return "Thanks for sharing these insights!";
    }
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
        return;
      }
      
      if (lowerLine.includes('quotes') || lowerLine.includes('notable quotes')) {
        currentSection = 'quotes';
        return;
      }
      
      // Parse key points (look for numbered points)
      if (currentSection === 'points') {
        const pointMatch = cleanLine.match(/^(\d+\.|\*|-)\s*(.+)$/);
        if (pointMatch && pointMatch[2]) {
          const point = pointMatch[2].trim();
          if (point.length > 10 && keyPoints.length < 3) {
            keyPoints.push(point.substring(0, 100));
          }
        }
        // Also catch lines that look like key points
        else if (cleanLine.length > 20 && cleanLine.length < 150 && !cleanLine.includes('quote') && keyPoints.length < 3) {
          keyPoints.push(cleanLine.substring(0, 100));
        }
      }
      
      // Parse quotes (look for "Quote 1:", "Quote 2:", etc.)
      else if (currentSection === 'quotes') {
        const quoteMatch = cleanLine.match(/^(quote\s*\d*:|")(.+)"?$/i);
        if (quoteMatch && quoteMatch[2]) {
          const quote = quoteMatch[2].trim().replace(/[""]/g, '');
          if (quote.length > 10 && quotes.length < 2) {
            quotes.push(quote.substring(0, 80));
          }
        }
        // Also catch lines that look like quotes
        else if (cleanLine.length > 15 && cleanLine.length < 120 && quotes.length < 2) {
          const potentialQuote = cleanLine.replace(/[""]/g, '').trim();
          if (potentialQuote.length > 15) {
            quotes.push(potentialQuote.substring(0, 80));
          }
        }
      }
    });

    // Enhanced fallback parsing
    if (keyPoints.length === 0 && quotes.length === 0) {
      console.log('⚠️ Using enhanced fallback parsing');
      
      // Look for any numbered or bulleted items
      lines.forEach(line => {
        const cleanLine = line.trim();
        if ((cleanLine.match(/^[\d*-]/) || cleanLine.length > 30) && cleanLine.length < 150) {
          const content = cleanLine.replace(/^[\d*-.]+\s*/, '').trim();
          if (content.length > 15) {
            if (keyPoints.length < 2) {
              keyPoints.push(content.substring(0, 100));
            } else if (quotes.length < 2) {
              quotes.push(content.replace(/[""]/g, '').substring(0, 80));
            }
          }
        }
      });
      
      // If still nothing, use first few meaningful lines
      if (keyPoints.length === 0 && quotes.length === 0) {
        const meaningfulLines = lines.filter(line => 
          line.trim().length > 20 && line.trim().length < 120
        );
        
        for (let i = 0; i < Math.min(3, meaningfulLines.length); i++) {
          if (i < 2) {
            keyPoints.push(meaningfulLines[i].trim().substring(0, 100));
          } else if (quotes.length < 2) {
            quotes.push(meaningfulLines[i].trim().replace(/[""]/g, '').substring(0, 80));
          }
        }
      }
    }

    const result = {
      keyPoints: keyPoints.length > 0 ? keyPoints : ["Main discussion points"],
      quotes: quotes.length > 0 ? quotes : ["Key statement from thread"],
      sentiment: sentiment,
      wordCount: 0,
      timeToRead: timeToRead || Math.max(1, Math.floor(lines.length / 50))
    };
    
    console.log('✅ Parsed summary result:', result);
    return result;
  }
}

module.exports = new GeminiService();
