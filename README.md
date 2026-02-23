# Steel Plant ML Prediction App

This is an industry-leading Steel Plant ML Prediction App built with React, Vite, and Tailwind CSS. It allows users to upload historical steel plant data, analyze it, train machine learning models, and predict optimal parameters.

## Features

-   **Onboarding Wizard**: A step-by-step guide to get started.
-   **Data Upload**: Drag-and-drop support for CSV and Excel files with automatic parsing and validation.
-   **Data Explorer**: Interactive dashboard with summary statistics, histograms, and correlation matrices.
-   **ML Training**: Train multiple models (Random Forest, Linear Regression, XGBoost simulation) directly in the browser/client using `ml-random-forest` and `ml-regression`.
-   **Predictions**: Use trained models to predict target variables based on input parameters, with confidence intervals.
-   **Theme**: Custom "Deep Steel Blue" and "Molten Orange" industrial theme.

## Tech Stack

-   **Frontend**: React 18, Vite, Tailwind CSS, Framer Motion
-   **UI Components**: shadcn/ui (Radix UI + Tailwind)
-   **Charts**: Recharts
-   **State Management**: Zustand (with persistence)
-   **ML Engine**: `ml-random-forest`, `ml-regression` (JavaScript implementations)
-   **Icons**: Lucide React

## Getting Started

1.  Install dependencies: `npm install`
2.  Start the development server: `npm run dev`
3.  Open the app in your browser.

## Usage Flow

1.  **Landing**: Click "Get Started" to enter the onboarding wizard.
2.  **Upload**: Upload a CSV or Excel file containing your steel plant data.
3.  **Explorer**: View the data distribution and correlations.
4.  **Training**: Select a target variable (e.g., Yield, Temperature) and train models.
5.  **Predictions**: Select the best trained model and enter input parameters to get a prediction.

## Note on ML Backend

Due to the environment constraints (WebContainer), the ML backend is implemented using JavaScript libraries (`ml-random-forest`, `ml-regression`) running in the client/Node environment instead of a separate Python FastAPI service. This ensures the application is fully functional and self-contained without requiring external Python infrastructure.
