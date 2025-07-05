# Multilingual QA Chatbot

This is a Streamlit application that allows users to ask questions in multiple languages and receive intelligent responses from a conversational AI agent. The chatbot uses semantic search to find relevant information from a knowledge base and provides answers in the user's preferred language. It features automatic language detection, multilingual support, and real-time translation capabilities.

## Features

- **Multilingual Support**: Ask questions in 7 different languages (English, Bahasa Melayu, Bengali, French, Spanish, German, Japanese)
- **Automatic Language Detection**: Automatically detects the language of your input question
- **Semantic Search**: Uses sentence transformers to find the most relevant information from the knowledge base
- **Question Answering**: Employs RoBERTa-based model for accurate answer extraction
- **Real-time Translation**: Translates responses to your preferred output language
- **Interactive Chat Interface**: Clean and intuitive chat-like user interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Multilingual_QA-Rag_based_Chatbot.git
```

2. Change to the project directory:

```bash
cd multilingual-qa-chatbot
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare your knowledge base(Optional) or use the existing data in the repo:
   - Use the `web_scrape.ipynb` to scrape wikipedia data in your preferred language, topics and truncation of characters as per choice
   - Data would be stored in `wiki_summaries.json` file in the `data` folder
   - The JSON file should contain an array of objects with `title` and `summary` fields:

```json
[
  {
    "title": "Artificial Intelligence",
    "summary": "Artificial intelligence (AI) is intelligence demonstrated by machines..."
  },
  ....
]
```

5. Run the Streamlit application:

```bash
streamlit run app.py
```

6. Open your web browser and go to `http://localhost:8501` to access the application.

## Usage

1. **Select Output Language**: Choose your preferred language for responses using the radio buttons
2. **Ask a Question**: Type your question in any supported language in the text input field
   [Caution] --> This is rag based system and using sentence transformers model so the knowledge is limited to the topics in the data
3. **Get Answers**: The chatbot will:
   - Automatically detect your input language as per supported languages
   - Search for relevant information in the knowledge base
   - Extract the most accurate answer using the QA model
   - Translate the response to your selected language
4. **View Chat History**: All conversations are displayed in a chat-like interface
5. **Clear Chat**: Use the "Clear Chat" button to start fresh
6. **Monitor Statistics**: Check the sidebar for document count and chat message statistics

## Supported Languages

| Language | Code | Native Name   |
| -------- | ---- | ------------- |
| English  | en   | English       |
| Malay    | ms   | Bahasa Melayu |
| Bengali  | bn   | বাংলা         |
| French   | fr   | Français      |
| Spanish  | es   | Español       |
| German   | de   | Deutsch       |
| Japanese | ja   | 日本語        |

## Technical Architecture

The application consists of several key components:

### Models Used

- **Sentence Transformer**: `all-MiniLM-L6-v2` for semantic search and document retrieval
- **Question Answering**: `deepset/roberta-base-squad2` for extracting answers from context
- **Language Detection**: `langdetect` library for automatic language identification
- **Translation**: `deep-translator` with Google Translate backend

### Core Components

- **Semantic Search**: Uses sentence embeddings to find the most relevant documents
- **Question Answering**: Extracts precise answers from retrieved document context
- **Language Processing**: Handles multilingual input/output with automatic translation
- **Chat Interface**: Streamlit-based conversational UI with message history

## Requirements

All the required library is in `requirements.txt` file.

## File Structure

```
multilingual-qa-chatbot
├── web_scrape.ipynb        # To scrape data from wikipedia
├── app.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/
    └── wiki_summaries.json # Knowledge base file
```

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request. Here are some ways you can contribute:

- Add support for more languages
- Improve the question answering accuracy
- Enhance the user interface
- Add more sophisticated retrieval methods
- Implement conversation memory
- Add export functionality for chat history

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This project uses several open-source libraries and models:

- **Streamlit**: For building the interactive web interface. Visit the [Streamlit documentation](https://docs.streamlit.io/) for more information.
- **Transformers**: For the RoBERTa-based question answering model. Visit the [Hugging Face Transformers](https://huggingface.co/transformers/) library for more information.
- **Sentence Transformers**: For semantic search capabilities. Visit the [Sentence Transformers](https://www.sbert.net/) documentation for more information.
- **Deep Translator**: For multilingual translation support. Visit the [Deep Translator](https://github.com/nidhaloff/deep-translator) GitHub repository for more information.
- **LangDetect**: For automatic language detection. Visit the [LangDetect](https://github.com/Mimino666/langdetect) GitHub repository for more information.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure you have a stable internet connection for downloading models on first run
2. **Translation Errors**: Check your internet connection as translation requires API access
3. **JSON Loading Errors**: Verify that your `wiki_summaries.json` file is properly formatted and located in the `data` folder
4. **Memory Issues**: For large knowledge bases, consider using a machine with more RAM or implementing batch processing

### Performance Tips

- Models are cached after first load for better performance
- Consider using GPU acceleration for faster inference if available
- For large knowledge bases, implement pagination or limit the number of documents loaded

## Future Enhancements

- Add support for document upload (PDF, DOCX, TXT)
- Implement conversation context memory
- Add voice input/output capabilities
- Create a REST API version
- Add user authentication and chat history persistence
- Implement custom model fine-tuning options
