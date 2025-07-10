# 🌍 AAROH - AI-Assisted Reasoning for Orchestrated Geospatial Handling

**AAROH** is an intelligent spatial analysis system that lets users ask natural language queries like

> “Show me the vegetation in Bengaluru?”
> and get real satellite-based visualizations with detailed step-by-step reasoning — powered by geospatial tools and LLMs.

This project is built as a submission for the ISRO AIML Hackathon.

---

## 🚀 Features

- 🔍 **Natural Language Understanding**: Converts user queries into GIS workflows.
- 🛰️ **Geospatial Analysis**: Supports flood risk, vegetation health, land cover, water detection, solar irradiance.
- 🧠 **LLM Reasoning Agent**: Uses Chain-of-Thought prompting with retrieval-based examples to select correct tools.
- 🗉 **Interactive Maps**: Streamlit + geemap to visualize results.
- 🔀 **Fallback Handling**: Switches to alternate datasets when primary data is missing.
- 📦 **Modular Codebase**: Easily extendable and tool-agnostic.

---

## 🧰 Technologies Used

| Component         | Stack                                 |
| ----------------- | ------------------------------------- |
| Frontend          | Streamlit, geemap, folium             |
| Geospatial Engine | Google Earth Engine (GEE)             |
| LLM Reasoning     | LangChain + Mistral (via Together.ai) |
| Data Retrieval    | Pandas, CSV (RAG-style examples)      |
| Visualization     | Leaflet via `geemap.foliumap`         |
| Environment       | Python (.env, dotenv)                 |

---

## 📁 Project Structure

```
AAROH/
│
├── Home.py               # Streamlit app UI and flow
│
├── gee/
│   └── flood.py          # GEE logic for flood, NDVI, water, solar, etc.
│
├── llm/
│   └── agent_tools.py    # Agent wrapper that chooses the right tool
│
├── rag/
│   └── retriever.py      # Retrieves similar queries using tags
│
├── examples.csv          # Example queries for retrieval (RAG base)
├── .env                  # API keys and secrets
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/TechSpire0/AAROH.git
cd AAROH

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys to `.env`
TOGETHER_API_KEY=your_together_api_key_here

# 5. Run the Streamlit app
streamlit run Home.py
```

> ✅ Make sure you have Google Earth Engine access and have authenticated with `ee.Authenticate()` at least once.

---

## 🎥 Demo

![AAROH Demo](https://github.com/TechSpire0/AAROH/assets/demo.gif)

---

## 👨‍💻 Contributors

- [Vishal](https://github.com/VishalK-1234)
- [Adapa Jayanth Kumar](https://github.com/jayanth131)
- [Priyanka Mahadasyam](https://github.com/Priyanka-Mahadasyam)
- [Satvika Mantri](https://github.com/satvika-mantri)

---

## 💬 Feedback

We’re improving this constantly. Feel free to:

- 🐛 Report bugs
- 💡 Suggest new tools or datasets
- 🌍 Share use cases
- ⭐ Star the repo if you find it useful!
  If you have any feedback, please reach us out at techspire000@gmail.com

---

> This project is built as a demo for the ISRO Bharatiya Antariksh Hackathon 2025 – with a mission to bring powerful spatial reasoning to everyone through natural language.
