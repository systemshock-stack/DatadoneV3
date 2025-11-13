"""
DatadoneV3 - Enterprise Sentiment Analytics Platform
Built for E: drive deployment with modern architecture
"""
import os
import sys
import io
import json
import re
import logging
import hashlib
import base64
import warnings
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from contextlib import contextmanager

# Set up paths for E: drive
BASE_DIR = Path("E:/DatadoneV3")
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models_cache"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories
for directory in [BASE_DIR, DATA_DIR, LOGS_DIR, UPLOADS_DIR, MODELS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = str(MODELS_DIR)

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, BLOB, Text, ForeignKey, Index, inspect
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.pool import StaticPool
import bcrypt

# ML Imports
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import spacy
from langdetect import detect, LangDetectException

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Security
from cryptography.fernet import Fernet

# Config
load_dotenv(BASE_DIR / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
DATABASE_URL = f"sqlite:///{DATA_DIR}/datadone.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database Models
class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    hashed_password = Column(BLOB, nullable=False)
    role = Column(String, nullable=False, default="analyst")
    company_id = Column(String, default="default")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Integer, default=1)

class LicenseKey(Base):
    __tablename__ = "license_keys"
    key = Column(String, primary_key=True)
    company_id = Column(String, nullable=False)
    tier = Column(String, default="professional")
    max_analyses = Column(Integer, default=1000)
    analyses_used = Column(Integer, default=0)
    expiry_date = Column(DateTime)
    is_active = Column(Integer, default=1)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    username = Column(String, ForeignKey("users.username"), index=True)
    company_id = Column(String)
    action = Column(String, nullable=False)
    details = Column(Text)
    ip_address = Column(String)

class UserPreference(Base):
    __tablename__ = "user_preferences"
    username = Column(String, ForeignKey("users.username"), primary_key=True)
    theme = Column(String, default="Light")
    default_model = Column(String, default="DistilBERT (English)")
    notification_enabled = Column(Integer, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database Manager
class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            db.close()
    
    def authenticate_user(self, username: str, password: str, license_key: str = None) -> Optional[Dict]:
        """Authenticate and check license"""
        with self.get_db() as db:
            company_id = "default"
            
            # If license provided, verify it and get company_id
            if license_key:
                license = db.query(LicenseKey).filter(
                    LicenseKey.key == license_key,
                    LicenseKey.is_active == 1
                ).first()
                if not license or license.expiry_date < datetime.utcnow():
                    return None
                company_id = license.company_id
            
            # Check user - try exact company first
            user = db.query(User).filter(
                User.username == username,
                User.company_id == company_id
            ).first()
            
            # Fallback: if no license entered and user not found, try demo company
            if not user and not license_key:
                user = db.query(User).filter(
                    User.username == username,
                    User.company_id == "demo-company"
                ).first()
            
            if user and bcrypt.checkpw(password.encode(), user.hashed_password):
                # Update last login
                user.last_login = datetime.utcnow()
                
                # Increment license usage
                if license_key and license:
                    license.analyses_used += 1
                
                return {
                    "username": user.username,
                    "role": user.role,
                    "company_id": company_id,
                    "license_key": license_key,
                    "theme": self.get_user_preference(username, "theme", "Light"),
                    "model": self.get_user_preference(username, "default_model", "DistilBERT (English)")
                }
        return None
    
    def get_user_preference(self, username: str, key: str, default: Any) -> Any:
        with self.get_db() as db:
            pref = db.query(UserPreference).filter(
                UserPreference.username == username
            ).first()
            if pref:
                return getattr(pref, key, default)
        return default
    
    def log_audit(self, username: str, action: str, details: str = None):
        with self.get_db() as db:
            log = AuditLog(
                username=username,
                company_id=st.session_state.get("company_id", "default"),
                action=action,
                details=details,
                ip_address=st.session_state.get("ip_address", "127.0.0.1")
            )
            db.add(log)

# License Manager
class LicenseManager:
    def __init__(self):
        self.tiers = {
            "starter": {"max_analyses": 100, "price": 49, "name": "Starter"},
            "professional": {"max_analyses": 1000, "price": 99, "name": "Professional"},
            "enterprise": {"max_analyses": 10000, "price": 299, "name": "Enterprise"}
        }
    
    def generate_key(self, tier: str, company_id: str) -> str:
        """Generate a license key"""
        if tier not in self.tiers:
            raise ValueError(f"Invalid tier: {tier}")
        
        prefix = f"DD3-{tier.upper()}-{datetime.now().year}"
        suffix = secrets.token_hex(6).upper()
        key = f"{prefix}-{suffix}"
        
        # Store in database
        db = SessionLocal()
        try:
            license = LicenseKey(
                key=key,
                company_id=company_id,
                tier=tier,
                max_analyses=self.tiers[tier]["max_analyses"],
                expiry_date=datetime.utcnow() + timedelta(days=365)
            )
            db.add(license)
            db.commit()
            return key
        finally:
            db.close()
    
    def validate(self, key: str) -> bool:
        """Check if license is valid"""
        db = SessionLocal()
        try:
            license = db.query(LicenseKey).filter(
                LicenseKey.key == key,
                LicenseKey.is_active == 1
            ).first()
            return license and license.expiry_date > datetime.utcnow()
        finally:
            db.close()
    
    def get_usage(self, company_id: str) -> Dict[str, Any]:
        """Get license usage statistics for a company"""
        db = SessionLocal()
        try:
            # Get the active license for the company
            license = db.query(LicenseKey).filter(
                LicenseKey.company_id == company_id,
                LicenseKey.is_active == 1
            ).first()
            
            if not license:
                return {"used": 0, "limit": 0, "percent": 0}
            
            return {
                "used": license.analyses_used,
                "limit": license.max_analyses,
                "percent": (license.analyses_used / license.max_analyses * 100) if license.max_analyses > 0 else 0
            }
        finally:
            db.close()

# Security Manager
class SecurityManager:
    def __init__(self):
        key_file = BASE_DIR / ".secret_key"
        if key_file.exists():
            self.key = key_file.read_bytes()
        else:
            self.key = Fernet.generate_key()
            key_file.write_bytes(self.key)
            key_file.chmod(0o600)
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, data: bytes) -> str:
        return self.cipher.decrypt(data).decode()

# Model Manager
class ModelManager:
    def __init__(self):
        self.device = self._get_device()
        self.models = {}
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @st.cache_resource
    def load_model(_self, model_name: str, model_type: str = "sentiment"):
        try:
            # Check torch version for security
            if torch.__version__ < "2.6.0":
                st.error("⚠️ Torch version must be >= 2.6.0 for security. Please upgrade: pip install torch>=2.6.0")
                return None
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODELS_DIR
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=MODELS_DIR
            )
            
            if model_type == "sentiment":
                return pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if _self.device == "cuda" else -1,
                    truncation=True,
                    max_length=512
                )
            return None
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            st.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def predict_sentiment(self, texts: List[str], model_name: str) -> List[Dict]:
        if model_name not in self.models:
            self.models[model_name] = self.load_model(model_name)
        
        model = self.models[model_name]
        if not model:
            # Return default values if model fails
            return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]
        
        results = []
        batch_size = 32
        
        with st.spinner("Analyzing sentiment..."):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_results = model(batch)
                    if not batch_results:
                        batch_results = [{"label": "NEUTRAL", "score": 0.0}] * len(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    results.extend([{"label": "NEUTRAL", "score": 0.0}] * len(batch))
        
        return results if results else [{"label": "NEUTRAL", "score": 0.0}] * len(texts)

# Data Processor
class DataProcessor:
    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        max_size = 100 * 1024 * 1024  # 100MB
        if file.size > max_size:
            return False, "File too large. Max size: 100MB"
        
        allowed = [".txt", ".csv", ".json", ".xlsx"]
        if not any(file.name.lower().endswith(ext) for ext in allowed):
            return False, f"Unsupported file type: {file.name}"
        
        return True, ""
    
    @staticmethod
    def load_file(file) -> Optional[pd.DataFrame]:
        try:
            file_io = io.BytesIO(file.getvalue())
            name = file.name.lower()
            
            if name.endswith(".csv"):
                return pd.read_csv(file_io, on_bad_lines="skip")
            elif name.endswith(".json"):
                return pd.read_json(file_io, lines=True)
            elif name.endswith(".xlsx"):
                return pd.read_excel(file_io)
            elif name.endswith(".txt"):
                lines = file_io.read().decode("utf-8", errors="ignore").splitlines()
                return pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})
            
            return None
        except Exception as e:
            logger.error(f"File loading error: {e}")
            st.error(f"Failed to load {file.name}: {e}")
            return None
    
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "text" not in df.columns:
            st.error("Data must contain a 'text' column")
            return pd.DataFrame()
        
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().astype(bool)]
        df = df.drop_duplicates(subset=["text"])
        
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT
        
        if len(df) > 10000:
            st.info("Dataset too large. Sampling 10,000 rows.")
            df = df.sample(10000, random_state=42)
        
        return df.reset_index(drop=True)

# Enterprise Analyzer (ALL features from v1)
class EnterpriseAnalyzer:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.nlp = None
    
    def analyze_sentiment(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        texts = df["text"].astype(str).tolist()
        results = self.model_manager.predict_sentiment(texts, model_name)
        
        if not results:
            results = [{"label": "NEUTRAL", "score": 0.0}] * len(df)
        
        df["label"] = [r.get("label", "NEUTRAL") for r in results]
        df["score"] = [r.get("score", 0.0) for r in results]
        
        def normalize(label, score):
            if "STAR" in str(label):
                try:
                    stars = int(str(label).split()[0])
                    return (stars - 3) / 2.0
                except:
                    return 0.0
            return score if "POSITIVE" in str(label) else -score
        
        df["compound"] = df.apply(lambda x: normalize(x["label"], x["score"]), axis=1)
        df["language"] = df["text"].apply(self._detect_language)
        
        return df
    
    @staticmethod
    def _detect_language(text: str) -> str:
        try:
            return detect(text[:200])
        except:
            return "unknown"
    
    def analyze_emotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add emotion detection"""
        try:
            emotion_model = self.model_manager.load_model(
                "j-hartmann/emotion-english-distilroberta-base",
                "emotion"
            )
            if emotion_model:
                texts = df["text"].astype(str).tolist()
                results = emotion_model(texts)
                df["emotions"] = results
                df["top_emotion"] = df["emotions"].apply(
                    lambda x: x[0]['label'] if x and isinstance(x, list) else 'neutral'
                )
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            st.warning("⚠️ Emotion model failed to load. Skipping emotion analysis.")
            df["top_emotion"] = "neutral"
        
        return df
    
    def extract_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self.nlp:
                self.nlp = spacy.load("en_core_web_sm")
            
            aspects = []
            for doc in self.nlp.pipe(df["text"].astype(str).tolist(), batch_size=32):
                aspects.append([
                    chunk.text for chunk in doc.noun_chunks
                    if chunk.root.pos_ in ["NOUN", "PROPN"]
                ][:15])
            
            df["aspects"] = aspects
        except Exception as e:
            logger.error(f"Aspect extraction failed: {e}")
            df["aspects"] = [[] for _ in range(len(df))]
        
        return df
    
    def generate_topics(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Topic modeling with TF-IDF + KMeans"""
        if len(df) < 10:
            return df
        
        sample_texts = df["text"].sample(min(1000, len(df))).astype(str).tolist()
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(sample_texts)
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(df) // 5),
            random_state=42,
            n_init=10
        )
        
        df['topic'] = kmeans.fit_predict(
            vectorizer.transform(df["text"].astype(str).tolist())
        )
        
        # Get top terms per topic
        feature_names = vectorizer.get_feature_names_out()
        topic_terms = {}
        for i in range(kmeans.n_clusters):
            top_indices = kmeans.cluster_centers_[i].argsort()[-10:][::-1]
            topic_terms[i] = ", ".join(feature_names[top_indices])
        
        df['topic_terms'] = df['topic'].map(topic_terms)
        return df
    
    def generate_wordcloud(self, df: pd.DataFrame) -> plt.Figure:
        text = " ".join(df["text"].astype(str).sample(min(5000, len(df)))).lower()
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'http\S+', '', text)
        
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=200
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig

# Main Application
class DatadoneV3App:
    def __init__(self):
        self.db = DatabaseManager()
        self.security = SecurityManager()
        self.model_manager = ModelManager()
        self.analyzer = EnterpriseAnalyzer(self.model_manager)
        self.data_processor = DataProcessor()
        self.license = LicenseManager()
        
        # Available models
        self.models = {
            "DistilBERT (English)": "distilbert-base-uncased-finetuned-sst-2-english",
            "BERT Multilingual": "nlptown/bert-base-multilingual-uncased-sentiment",
            "RoBERTa (Advanced)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        }
        
        self.init_session()
    
    def init_session(self):
        defaults = {
            "authenticated": False,
            "username": None,
            "role": None,
            "company_id": "default",
            "license_key": None,
            "theme": "Light",
            "selected_model": "DistilBERT (English)",
            "demo_mode": True
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def login_page(self):
        """Login with license key"""
        st.title("🔐 DatadoneV3 Enterprise Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Demo mode checkbox
            is_demo = st.checkbox("🔓 Use Demo Mode (No License)", value=True)
            
            license_key = st.text_input("License Key (optional)", type="password", disabled=is_demo)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True):
                if is_demo:
                    # Demo mode - bypass license check
                    if username == "analyst" and password == "pro123":
                        st.session_state.authenticated = True
                        st.session_state.username = "analyst"
                        st.session_state.role = "admin"
                        st.session_state.company_id = "demo-company"
                        st.session_state.license_key = "DD3-DEMO-2025-00000000"
                        st.rerun()
                    else:
                        st.error("❌ Demo credentials: analyst / pro123")
                        self.db.log_audit(username, "Failed demo login")
                else:
                    # Full authentication with license
                    user = self.db.authenticate_user(username, password, license_key)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = user["username"]
                        st.session_state.role = user["role"]
                        st.session_state.company_id = user["company_id"]
                        st.session_state.license_key = user["license_key"]
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials or expired license")
                        self.db.log_audit(username, "Failed login attempt")
            
            st.info("📝 Demo: Username: **analyst** | Password: **pro123**")
    
    def sidebar_controls(self):
        """Enterprise sidebar"""
        st.sidebar.title("⚙️ Enterprise Configuration")
        
        # Show company info
        if st.session_state.company_id != "default":
            st.sidebar.success(f"🔑 Licensed to: {st.session_state.company_id}")
        
        # Model selection
        model_names = list(self.models.keys())
        st.session_state.selected_model = st.sidebar.selectbox(
            "Sentiment Model",
            model_names,
            index=model_names.index(st.session_state.selected_model)
        )
        
        # Features
        st.sidebar.header("Analysis Options")
        features = {
            "extract_aspects": st.sidebar.checkbox("Extract Aspects", True),
            "analyze_emotions": st.sidebar.checkbox("Emotion Detection", True),
            "generate_topics": st.sidebar.checkbox("Topic Modeling", False),
            "generate_wordcloud": st.sidebar.checkbox("Generate Word Cloud", True),
            "anonymize_pii": st.sidebar.checkbox("Anonymize PII", False)
        }
        
        # Theme
        st.session_state.theme = st.sidebar.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1
        )
        
        # Admin panel
        if st.session_state.role == "admin":
            with st.sidebar.expander("🔧 Admin Panel"):
                if st.button("Generate License Key"):
                    self.show_license_generator()
        
        return features
    
    def show_license_generator(self):
        """License generation UI for admins"""
        st.subheader("Generate New License Key")
        
        tier = st.selectbox("Tier", ["starter", "professional", "enterprise"])
        company_id = st.text_input("Company ID")
        
        if st.button("Generate"):
            if company_id:
                key = self.license.generate_key(tier, company_id)
                st.success(f"License Key Generated: `{key}`")
                st.warning("⚠️ Copy this key now - it won't be shown again!")
                self.db.log_audit(st.session_state.username, f"Generated license for {company_id}")
            else:
                st.error("Company ID required")
    
    def upload_section(self) -> Optional[pd.DataFrame]:
        """File upload with progress"""
        st.header("📁 Data Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Files (CSV, JSON, Excel, TXT)",
            type=["csv", "json", "xlsx", "txt"],
            accept_multiple_files=True
        )
        
        if not uploaded_files:
            return None
        
        dfs = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            is_valid, msg = self.data_processor.validate_file(file)
            if not is_valid:
                st.error(f"❌ {file.name}: {msg}")
                continue
            
            df = self.data_processor.load_file(file)
            if df is not None:
                dfs.append(df)
                st.success(f"✅ Loaded {file.name}: {len(df)} rows")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        progress_bar.empty()
        
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        processed_df = self.data_processor.preprocess(combined_df)
        
        self.db.log_audit(
            st.session_state.username,
            "Data uploaded",
            f"{len(processed_df)} records from {len(uploaded_files)} files"
        )
        
        return processed_df
    
    def dashboard(self, df: pd.DataFrame, features: Dict):
        """Main dashboard with all features"""
        st.title("📊 DatadoneV3 Enterprise Dashboard")
        
        # Show license usage (only if not demo mode)
        if st.session_state.license_key and st.session_state.company_id != "demo-company":
            try:
                usage = self.license.get_usage(st.session_state.company_id)
                if usage['limit'] > 0:
                    st.sidebar.progress(
                        usage['percent'] / 100,
                        f"License Usage: {usage['used']}/{usage['limit']} analyses"
                    )
            except Exception as e:
                logger.error(f"Failed to get usage: {e}")
        
        # Metrics
        self.show_metrics(df)
        
        # Run analysis
        model_name = self.models[st.session_state.selected_model]
        df = self.analyzer.analyze_sentiment(df, model_name)
        
        if features["analyze_emotions"]:
            df = self.analyzer.analyze_emotions(df)
        
        if features["extract_aspects"]:
            df = self.analyzer.extract_aspects(df)
        
        if features["generate_topics"] and len(df) > 10:
            df = self.analyzer.generate_topics(df)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.bar(
                df["label"].value_counts().reset_index(),
                x="label", y="count", color="label"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if "top_emotion" in df.columns:
                st.subheader("Emotion Distribution")
                fig = px.pie(
                    df["top_emotion"].value_counts().reset_index(),
                    values="count", names="top_emotion"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Avg Sentiment by Label")
            fig = px.bar(
                df.groupby("label")["compound"].mean().reset_index(),
                x="label", y="compound", color="label"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if "topic" in df.columns:
                st.subheader("Topic Distribution")
                topic_df = df["topic"].value_counts().reset_index()
                topic_df["topic_terms"] = topic_df["topic"].map(
                    df.set_index("topic")["topic_terms"].to_dict()
                )
                fig = px.bar(topic_df, x="topic", y="count", hover_data=["topic_terms"])
                st.plotly_chart(fig, use_container_width=True)
        
        # Word Cloud
        if features["generate_wordcloud"]:
            st.subheader("☁️ Word Cloud")
            fig = self.analyzer.generate_wordcloud(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Data Table
        st.subheader("📋 Data Preview")
        display_cols = ["text", "label", "score", "compound", "language"]
        if "top_emotion" in df.columns:
            display_cols.append("top_emotion")
        if "aspects" in df.columns:
            display_cols.append("aspects")
        if "topic" in df.columns:
            display_cols.append("topic")
        
        st.dataframe(
            df[display_cols].head(1000),
            use_container_width=True,
            height=400
        )
        
        # Export
        self.export_section(df)
    
    def show_metrics(self, df: pd.DataFrame):
        """Display KPI metrics"""
        metrics = st.columns(4)
        
        with metrics[0]:
            st.metric("Total Records", f"{len(df):,}")
        
        with metrics[1]:
            avg_sentiment = df["text"].apply(len).mean() if not df.empty else 0
            st.metric("Avg Text Length", f"{avg_sentiment:.0f}")
        
        with metrics[2]:
            languages = df.get("language", pd.Series(["N/A"])).nunique()
            st.metric("Languages", languages)
        
        with metrics[3]:
            if "compound" in df.columns:
                avg_compound = df["compound"].mean()
                st.metric("Avg Sentiment", f"{avg_compound:.2f}")
    
    def export_section(self, df: pd.DataFrame):
        """Export functionality"""
        st.header("📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV Export
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "sentiment_analysis.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Excel Export
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Analysis")
            st.download_button(
                "Download Excel",
                buffer.getvalue(),
                "sentiment_analysis.xlsx",
                "application/vnd.ms-excel",
                use_container_width=True
            )
        
        # JSON Export
        with col3:
            json_data = df.to_json(orient="records", indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                "sentiment_analysis.json",
                "application/json",
                use_container_width=True
            )
    
    def run(self):
        """Main application loop"""
        st.set_page_config(
            page_title="DatadoneV3 Enterprise",
            page_icon="📊",
            layout="wide"
        )
        
        # Check authentication
        if not st.session_state.authenticated:
            self.login_page()
            return
        
        # Show logout and user info
        st.sidebar.info(f"👤 Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.sidebar.button("🚪 Logout"):
            self.db.log_audit(st.session_state.username, "User logged out")
            st.session_state.clear()
            st.rerun()
        
        # Sidebar controls
        features = self.sidebar_controls()
        
        # Upload data
        df = self.upload_section()
        
        if df is not None and not df.empty:
            # Run analysis and show dashboard
            self.dashboard(df, features)
            
            # Log completion
            self.db.log_audit(
                st.session_state.username,
                "Analysis completed",
                f"Processed {len(df)} records"
            )

# Global function for setup script
def init_db():
    """Initialize database - called by setup.bat"""
    Base.metadata.create_all(engine)
    
    # Add demo user and license
    db = SessionLocal()
    try:
        # Demo license
        demo_license = LicenseKey(
            key="DD3-DEMO-2025-00000000",
            company_id="demo-company",
            tier="enterprise",
            max_analyses=999999,
            analyses_used=0,
            expiry_date=datetime.utcnow() + timedelta(days=365),
            is_active=1
        )
        db.merge(demo_license)
        
        # Demo user
        demo_pass = os.getenv("DEMO_PASSWORD", "pro123").encode()
        hashed = bcrypt.hashpw(demo_pass, bcrypt.gensalt())
        
        user = db.query(User).filter(User.username == "analyst").first()
        if not user:
            user = User(
                username="analyst",
                hashed_password=hashed,
                role="admin",
                company_id="demo-company"
            )
            db.add(user)
            db.commit()
            logger.info("Demo user created")
            print("✅ Database initialized successfully!")
        else:
            print("ℹ️ Database already exists.")
    finally:
        db.close()
    print("✅ Setup complete!")

if __name__ == "__main__":
    # Check if database exists, if not initialize it
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        logger.info("Database not found, initializing...")
        init_db()
    
    app = DatadoneV3App()
    app.run()