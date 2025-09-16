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
- Provide EXACTLY 3 key points (no more, no less)
- Include EXACTLY 2 notable quotes (most impactful statements)
- Keep each key point under 100 characters
- Keep each quote under 80 characters
- Focus on the main discussion, ignore replies/suggestions
- Use clear, simple language
- Do not include markdown or special formatting

Thread content: ${threadContent.text.substring(0, 2500)}

Format your response as clean text with clear sections. No markdown, no extra formatting.`;

      return await this.retryOperation(() => this.callGeminiAPI(prompt));
    } catch (error) {
      throw new Error(`Gemini API error: ${error.message}`);
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

      return await this.retryOperation(() => this.callGeminiAPI(prompt));
    } catch (error) {
      throw new Error(`Gemini API error: ${error.message}`);
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
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
    
    throw lastError;
  }

  parseSummaryResponse(aiResponse) {
    const lines = aiResponse.split('\n').filter(line => line.trim());
    
    const keyPoints = [];
    const quotes = [];
    let sentiment = "neutral";
    let timeToRead = 1;

    let currentSection = '';
    lines.forEach(line => {
      const lowerLine = line.toLowerCase();
      
      if (lowerLine.includes('key points') || lowerLine.includes('main points')) {
        currentSection = 'points';
      } else if (lowerLine.includes('notable quotes') || lowerLine.includes('quotes')) {
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
          if (keyPoints.length < 3) { // Limit to 3 key points
            keyPoints.push(cleanLine.substring(0, 100));
          }
        } else if (currentSection === 'quotes' && cleanLine) {
          if (quotes.length < 2) { // Limit to 2 quotes
            quotes.push(cleanLine.replace(/[""]/g, '').substring(0, 80));
          }
        }
      }
    });

    // Fallback parsing with strict limits
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
