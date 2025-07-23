# Baggage ML Prediction System

## Overview

This is a Flask-based web application that provides machine learning predictions for various baggage handling scenarios in airports. The system uses trained XGBoost models to predict risks related to baggage damage, system downtime, departure delays, mishandled baggage, and transfer SLA compliance. The application features a dark-themed Bootstrap interface with interactive prediction forms and performance dashboards.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript Libraries**: Chart.js for data visualization, Feather Icons for iconography
- **Styling**: Custom CSS with Bootstrap integration and smooth animations
- **Responsive Design**: Mobile-first approach with responsive layouts

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Architecture Pattern**: MVC-like structure with routes handling business logic
- **Model Management**: Centralized ModelPredictor class for loading and managing ML models
- **Error Handling**: Comprehensive logging and error management
- **Development Server**: Built-in Flask development server with hot reload

### Machine Learning Pipeline
- **ML Framework**: XGBoost for classification models
- **Preprocessing**: StandardScaler for feature scaling, LabelEncoder for categorical variables
- **Model Types**: Five specialized prediction models (damage, downtime, departure, mishandled, transfer)
- **Model Persistence**: Joblib for serializing trained models and preprocessors
- **Training Pipeline**: Automated training script with data preprocessing and model evaluation

## Key Components

### Core Application Structure
- `app.py`: Flask application factory with configuration and middleware setup
- `main.py`: Application entry point for production deployment
- `routes.py`: Request routing and business logic (incomplete in provided files)
- `train_models.py`: ML model training pipeline and data preprocessing

### Model Management System
- **ModelPredictor Class**: Centralized model loading and prediction interface
- **Model Storage**: File-based model persistence in `/models` directory
- **Feature Engineering**: Automated preprocessing pipelines for each prediction type
- **Performance Tracking**: Model metrics storage and retrieval system

### User Interface Components
- **Base Template**: Unified navigation and layout structure
- **Prediction Forms**: Specialized input forms for each prediction type
- **Dashboard Views**: Model performance visualization and status monitoring
- **Responsive Design**: Mobile-optimized interface with Bootstrap components

### Prediction Modules
1. **Damage Risk**: Predicts baggage damage likelihood based on flight characteristics
2. **System Downtime**: Forecasts potential system failures and maintenance needs
3. **Departure Delays**: Assesses risk of baggage-related flight delays
4. **Mishandled Baggage**: Identifies flights at risk of baggage mishandling
5. **Transfer SLA**: Monitors compliance risk for connecting flight baggage transfers

## Data Flow

### Training Data Pipeline
1. CSV data ingestion from `attached_assets` directory
2. Data preprocessing and feature engineering for flight-level aggregation
3. Model training with cross-validation and hyperparameter optimization
4. Model serialization and performance metrics storage
5. Feature importance analysis and model validation

### Prediction Pipeline
1. User input collection through web forms
2. Input validation and preprocessing using stored scalers/encoders
3. Model inference using trained XGBoost classifiers
4. Risk assessment and confidence scoring
5. Results visualization with charts and risk indicators

### Model Performance Monitoring
1. Real-time model status checking and health monitoring
2. Performance metrics display (accuracy, precision, recall, F1-score)
3. Feature importance visualization for model interpretability
4. Historical performance tracking and trend analysis

## External Dependencies

### Python Libraries
- **Flask**: Web framework and routing
- **XGBoost**: Machine learning model implementation
- **Scikit-learn**: Data preprocessing and model evaluation
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Joblib**: Model serialization and persistence

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme support
- **Chart.js**: Interactive data visualization and charting
- **Feather Icons**: Lightweight icon library for consistent UI
- **Custom CSS**: Enhanced styling with animations and transitions

### Development Tools
- **Werkzeug ProxyFix**: Proxy handling for deployment environments
- **Python Logging**: Comprehensive application logging and debugging
- **Flask Debug Mode**: Development server with auto-reload capabilities

## Deployment Strategy

### Development Environment
- Flask development server with debug mode enabled
- Hot reload functionality for rapid development iteration
- Local file-based model storage and static asset serving
- Environment variable configuration for sensitive settings

### Production Considerations
- ProxyFix middleware configured for reverse proxy deployment
- Session management with configurable secret keys
- Error handling and logging for production monitoring
- Static file serving optimization for web assets

### Scalability Features
- Modular model loading system for easy model updates
- Centralized configuration management through environment variables
- Stateless prediction API design for horizontal scaling
- File-based model storage with potential for cloud storage migration

### Security Measures
- Session secret key management through environment variables
- Input validation and sanitization for all user inputs
- CSRF protection through Flask's built-in security features
- Secure handling of model files and sensitive data