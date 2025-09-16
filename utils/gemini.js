const { GoogleGenerativeAI } = require("@google/generative-ai");

class GeminiService {
  constructor() {
    this.genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    // Use the correct model name
    this.model = this.genAI.getGenerativeModel({ 
      model: process.env.GEMINI_MODEL || "gemini-1.5-flash" 
    });
  }

  async summarizeThread(threadContent) {
    try {
      const prompt = `Please analyze this thread and provide:
1. 3-5 key points (concise, actionable insights)
2. 2-3 notable quotes (most impactful statements)
3. Overall sentiment (positive/negative/neutral)
4. Estimated reading time in minutes

Thread content: ${threadContent.text.substring(0, 3000)}`;

      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      return this.parseSummaryResponse(text);
    } catch (error) {
      if (error.message.includes('429')) {
        throw new Error('API rate limit exceeded. Please try again later.');
      }
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
- Is 1-2 sentences maximum

Thread content: ${threadContent.text.substring(0, 2000)}

Summary of key points: ${summary.keyPoints.join(', ')}

Generate only the response text, nothing else:`;

      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text().trim();
    } catch (error) {
      if (error.message.includes('429')) {
        throw new Error('API rate limit exceeded. Please try again later.');
      }
      throw new Error(`Gemini API error: ${error.message}`);
    }
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
          keyPoints.push(cleanLine);
        } else if (currentSection === 'quotes' && cleanLine) {
          quotes.push(cleanLine.replace(/[""]/g, ''));
        }
      }
    });

    // Fallback parsing if structured parsing didn't work
    if (keyPoints.length === 0 && quotes.length === 0) {
      lines.forEach(line => {
        if (line.match(/^[\d*-]/) || line.length > 30) {
          const cleanLine = line.replace(/^[\d*-.]+\s*/, '').trim();
          if (cleanLine.length > 20) {
            if (keyPoints.length < 3) {
              keyPoints.push(cleanLine.substring(0, 150));
            } else if (quotes.length < 2) {
              quotes.push(cleanLine.substring(0, 100));
            }
          }
        }
      });
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 5) : ["Key insights from the thread"],
      quotes: quotes.length > 0 ? quotes.slice(0, 3) : ["Notable statements from the discussion"],
      sentiment: sentiment,
      wordCount: 0,
      timeToRead: timeToRead || Math.max(1, Math.floor(lines.length / 50))
    };
  }
}

module.exports = new GeminiService();
