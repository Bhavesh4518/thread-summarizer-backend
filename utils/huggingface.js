const { HfInference } = require('@huggingface/inference');

class HuggingFaceService {
  constructor() {
    this.HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
    if (!this.HF_TOKEN) {
      console.warn('⚠️ Hugging Face API key not found in environment variables');
    }
    this.inference = new HfInference(this.HF_TOKEN);
    
    // Default models - you can change these based on your testing
    this.SUMMARIZATION_MODEL = 'facebook/bart-large-cnn'; // Good for summarization
    this.REPLY_MODEL = 'microsoft/DialoGPT-medium'; // Good for conversational replies
  }

  async summarizeThread(threadContent) {
    try {
      console.log('🤖 Calling Hugging Face for summarization...');
      
      // Prepare the prompt for summarization
      const prompt = `Summarize this social media thread. Provide exactly 3 key points and 2 notable quotes.
      
Thread content: ${threadContent.text.substring(0, 1000)}

Format your response as:
Key Points:
1. [First key point]
2. [Second key point]  
3. [Third key point]

Quotes:
1. [First notable quote]
2. [Second notable quote]`;

      // Use text generation model for more control
      const response = await this.inference.textGeneration({
        model: 'google/flan-t5-base', // Instruction-following model
        inputs: prompt,
        parameters: {
          max_new_tokens: 200,
          temperature: 0.7,
          top_p: 0.9,
          repetition_penalty: 1.2
        }
      });

      console.log('✅ Hugging Face summary response:', response);
      
      // Parse the response
      return this.parseSummaryResponse(response.generated_text);
    } catch (error) {
      console.error('💥 Hugging Face summarization error:', error);
      
      // Fallback to basic summary if HF fails
      throw new Error(`Hugging Face API error: ${error.message}`);
    }
  }

  async generateReply(threadContent, summary) {
    try {
      console.log('🤖 Calling Hugging Face for reply generation...');
      
      const prompt = `Generate a human-like response to this social media thread:
      
Thread content: ${threadContent.text.substring(0, 500)}
Summary key points: ${summary.keyPoints.slice(0, 2).join(', ')}

Generate only a concise, natural response (1-2 sentences):`;

      const response = await this.inference.textGeneration({
        model: 'google/flan-t5-base',
        inputs: prompt,
        parameters: {
          max_new_tokens: 100,
          temperature: 0.8,
          top_p: 0.9
        }
      });

      console.log('✅ Hugging Face reply response:', response);
      
      return response.generated_text.trim();
    } catch (error) {
      console.error('💥 Hugging Face reply error:', error);
      throw new Error(`Hugging Face API error: ${error.message}`);
    }
  }

  parseSummaryResponse(aiResponse) {
    console.log('🔍 Parsing Hugging Face response:', aiResponse.substring(0, 100) + '...');
    
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
      console.log('⚠️ Using fallback parsing for Hugging Face response');
      const cleanLines = lines.filter(line => line.length > 20 && line.length < 150);
      for (let i = 0; i < Math.min(3, cleanLines.length); i++) {
        if (keyPoints.length < 2) {
          keyPoints.push(cleanLines[i].substring(0, 100));
        } else if (quotes.length < 2) {
          quotes.push(cleanLines[i].substring(0, 80));
        }
      }
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints.slice(0, 3) : ["Key insights from the thread"],
      quotes: quotes.length > 0 ? quotes.slice(0, 2) : ["Notable statements from the discussion"],
      sentiment: sentiment,
      wordCount: 0,
      timeToRead: timeToRead || Math.max(1, Math.floor(lines.length / 50))
    };
  }
}

module.exports = new HuggingFaceService();
