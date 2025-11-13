1. E:
2. cd \DatadoneV3
3. pip install -r requirements.txt
4. python -m spacy download en_core_web_sm
5. python -c "from DatadoneV3 import init_db; init_db()"
6. streamlit run DatadoneV3.py