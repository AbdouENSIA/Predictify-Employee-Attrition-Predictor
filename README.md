# 🚀 Employee Attrition Predictor: Unveiling Workplace Mysteries! 🕵️‍♂️

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Model Training](#7-model-training)
8. [Model Performance](#8-model-performance)
9. [Visualizations](#9-visualizations)
10. [Future Improvements](#10-future-improvements)
11. [Contributing](#11-contributing)
12. [License](#12-license)

## 1. Overview

Welcome to the Employee Attrition Predictor, where we turn HR mysteries into data-driven insights! 🎩✨

This project uses machine learning to predict employee attrition, helping companies understand why employees might decide to pack up their desk plants and bid farewell. We're using the power of Random Forests 🌳 (no actual forests were harmed in the making of this model) and the magic of Optuna 🔮 to create a prediction model that would make even Sherlock Holmes jealous.

Our goal? To help you keep your star employees shooting for the stars... at your company, not the competitor's! 

But what exactly is employee attrition, you ask? It's the rate at which employees leave a company and need to be replaced. High attrition can be costly and disruptive, so predicting and understanding it can be incredibly valuable for businesses.

## 2. Dataset

Welcome to the enchanted realm of data, where our crystal ball (a.k.a. the dataset) is brought to you by the illustrious IBM Watson Analytics Lab. It’s almost like they had a psychic connection with our needs—or maybe they just had a lot of exit interviews. 🔮✨

Our dataset, `employee_attrition.csv`, is packed with a treasure trove of employee details. Here’s a sneak peek into our magical repository:

| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| Age | 🎂 How many candles on their birthday cake | Numeric | 41 |
| Attrition | 🚪 Did they wave goodbye to the company? (target variable) | Categorical | Yes/No |
| BusinessTravel | ✈️ How often they pack their suitcase | Categorical | Travel_Rarely, Travel_Frequently, Non-Travel |
| DailyRate | 💵 How much they earn per day (ka-ching!) | Numeric | 1102 |
| Department | 🏢 Where they hang their hat at work | Categorical | Sales, Research & Development, Human Resources |
| DistanceFromHome | 🏠 How many miles they commute | Numeric | 1 |
| Education | 🎓 Their academic adventure (1-5) | Numeric | 2 |
| EducationField | 📚 Their field of study | Categorical | Life Sciences, Medical, Marketing |
| EmployeeCount | 👥 Number of employees (always 1, a bit redundant) | Numeric | 1 |
| EmployeeNumber | 🔢 Their unique badge number | Numeric | 1 |
| EnvironmentSatisfaction | 😊 How happy they are with their workspace | Numeric | 2 |
| Gender | 🚻 Their gender | Categorical | Female, Male |
| HourlyRate | 💰 How much they earn per hour | Numeric | 94 |
| JobInvolvement | 🛠️ How involved they are in their job (1-4) | Numeric | 3 |
| JobLevel | 🏆 Their job rank (1-5) | Numeric | 2 |
| JobRole | 🧑‍💼 Their role in the company | Categorical | Sales Executive, Research Scientist |
| JobSatisfaction | 😃 How satisfied they are with their job (1-4) | Numeric | 4 |
| MaritalStatus | 💍 Their marital status | Categorical | Single, Married, Divorced |
| MonthlyIncome | 💵 How much they earn per month | Numeric | 5993 |
| MonthlyRate | 💸 Monthly rate | Numeric | 19479 |
| NumCompaniesWorked | 🏢 How many companies they’ve worked at | Numeric | 8 |
| Over18 | 🎫 If they’re over 18 (always 'Y', not used in modeling) | Categorical | Y |
| OverTime | ⏳ If they work overtime | Categorical | Yes/No |
| PercentSalaryHike | 📈 Percent increase in salary last year | Numeric | 11 |
| PerformanceRating | 🌟 Performance rating (1-4) | Numeric | 3 |
| RelationshipSatisfaction | 💑 How satisfied they are with workplace relationships (1-4) | Numeric | 1 |
| StandardHours | ⏰ Standard hours (always 80, not used in modeling) | Numeric | 80 |
| StockOptionLevel | 📊 Stock option level (0-3) | Numeric | 0 |
| TotalWorkingYears | ⏳ Total years worked | Numeric | 8 |
| TrainingTimesLastYear | 📚 Hours spent training last year | Numeric | 0 |
| WorkLifeBalance | ⚖️ Work-life balance rating (1-4) | Numeric | 1 |
| YearsAtCompany | 🏢 Years at the company | Numeric | 6 |
| YearsInCurrentRole | 🕰️ Years in current role | Numeric | 4 |
| YearsSinceLastPromotion | 📅 Years since last promotion | Numeric | 0 |
| YearsWithCurrManager | 👩‍💼 Years with current manager | Numeric | 5 |

With 35 features to explore, we’re not just predicting attrition; we’re crafting employee biographies! 📚✨

The target variable here is 'Attrition'. It’s a binary variable where 'Yes' means the employee has bid farewell to the company, and 'No' means they’re still enjoying their coffee breaks at their desk. ☕🚀

## 3. Project Structure

Our project is as well-organized as Marie Kondo's sock drawer. Here's the lay of the land:

```
employee_attrition_predictor/
│
├── employee_attrition.csv
├── employee_attrition.ipynb
├── model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
```

Let's break it down:

- `employee_attrition.csv`: Our dataset, the treasure map to employee attrition.
- `employee_attrition.ipynb`: The Jupyter notebook where all the magic happens. It's like our spellbook, but with more Python and fewer eye of newt.
- `model.pkl`: Our trained model in pickle format. It's like we've bottled our attrition-predicting potion for easy use later!
- `requirements.txt`: The guest list for our Python package party.
- `README.md`: You are here! 👋
- `LICENSE`: The rules of the game.

## 4. Installation

Time to set up our crystal ball! Follow these steps:

1. Clone this repo faster than you can say "I quit!":
   ```
   git clone https://github.com/AbdouENSIA/employee-attrition-predictor.git
   cd employee-attrition-predictor
   ```

2. Create a virtual environment (because we like to keep things tidy):
   ```
   python -m venv attrition_env
   source attrition_env/bin/activate  # On Windows, use `attrition_env\Scripts\activate`
   ```

3. Install the required packages (it's like assembling your data science Avengers):
   ```
   pip install -r requirements.txt
   ```

   Here's what you're getting:
   - pandas (2.2.2): Your data manipulation Swiss Army knife
   - numpy (2.1.1): For when you need to crunch numbers like a boss
   - scikit-learn (1.5.1): Machine learning algorithms galore
   - matplotlib (3.9.2) and seaborn (0.13.2): For charts that make your eyes happy
   - optuna (4.0.0): Your personal hyperparameter optimization genie
   - jupyterlab (4.2.5): Because who doesn't love a good notebook?

   Want to make sure you're up-to-date with the latest versions? Run:
   ```
   pip install --upgrade pandas numpy scikit-learn matplotlib seaborn optuna jupyterlab
   ```
   But be warned: with great power comes great responsibility. And sometimes, incompatibility issues. Always test after upgrading!

## 5. Usage

Ready to predict the future? Here's how:

1. Fire up Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `employee_attrition.ipynb`.

3. Run the cells one by one, or select "Run All" if you're feeling adventurous.

4. Watch in awe as your computer turns into a fortune-telling machine for employee retention!

Want to use the pre-trained model? It's as easy as pie (mmmm, pie 🥧):

```python
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(your_data)
```

Just make sure `your_data` is preprocessed the same way as in the notebook. Our model isn't a mind reader... yet.

## 6. Data Preprocessing

Before we can turn our data into predictions, we need to give it a makeover. Here's what we do:

1. **Handling Missing Values**: We check for any employees who mysteriously disappeared from our dataset (missing values). In this dataset, we're lucky – no missing values! But if there were, we'd decide whether to fill in the blanks (imputation) or bid those rows farewell (deletion).

2. **Encoding Categorical Variables**: We turn categories into numbers, because our model prefers math to words. We use two methods:
   - Label Encoding: For ordinal categories (like Education Level)
   - One-Hot Encoding: For nominal categories (like Department)
   
   For example, we might turn "Department" into:
   - Sales: [1, 0, 0]
   - Research & Development: [0, 1, 0]
   - Human Resources: [0, 0, 1]

3. **Feature Scaling**: We make sure all our numerical features are on the same playing field. We use StandardScaler, which makes the mean of each feature 0 and scales it to unit variance. This way, 'Age' (0-100) doesn't overshadow 'JobSatisfaction' (1-4)!

4. **Handling Imbalanced Data**: In our dataset, most employees aren't leaving (thank goodness!). But this means our classes are imbalanced. We use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class (employees who left), ensuring our model learns to predict both outcomes equally well.

## 7. Model Training

Now for the main event! We're using a Random Forest Classifier, which is like having a whole forest of decision trees voting on whether an employee is likely to leave. 

Here's the process:

1. **Split the Data**: We divide our data into training (80%) and testing (20%) sets. It's like teaching our model with flashcards, then giving it a pop quiz.

2. **Initial Model Training**: We start with a basic Random Forest model. It's not perfect, but it's honest work.

3. **Hyperparameter Tuning**: This is where Optuna comes in. It's like having a personal trainer for our model, trying out different combinations of:
   - `n_estimators`: Number of trees in the forest
   - `max_depth`: Maximum depth of the trees
   - `min_samples_split`: Minimum number of samples required to split an internal node
   - `min_samples_leaf`: Minimum number of samples required to be at a leaf node

   Optuna uses a Bayesian optimization approach, learning from each trial to make smarter choices in the next one.

4. **Final Model Training**: We take the best hyperparameters Optuna found and train our final model. It's like the final form of a Pokemon, but for predicting employee attrition.

5. **Model Persistence**: We save our trained model using pickle. This way, we can use our model later without having to retrain it every time. It's like freezing a gourmet meal for later!

## 8. Model Performance

Time to see how our crystal ball performs! We use these metrics to evaluate our model:

- **Mean Absolute Error (MAE)**: Average magnitude of errors in predictions, without considering their direction.
- **Mean Squared Error (MSE)**: Average of the squares of errors, penalizing larger errors more severely.
- **Root Mean Squared Error (RMSE)**: Square root of the MSE, representing the standard deviation of prediction errors.
- **R² Score**: The proportion of variance in the dependent variable that is predictable from the independent variables.

We'll present these metrics in a neat table:

| Metric | Score |
|--------|-------|
| MAE    | 0.15  |
| MSE    | 0.15  |
| RMSE   | 0.39  |
| R²     | 0.18  |

Remember, in the world of employee attrition, a high recall might be more important than high precision. After all, it's better to give a stay-incentive to someone who wasn't going to leave than to lose someone because we didn't see it coming!

## 9. Visualizations

Because a picture is worth a thousand spreadsheets, we create:

1. **Feature Importance Plot**: See which factors are the MVPs in predicting attrition. This bar chart displays the importance of each feature in our Random Forest model.

2. **Optuna Study History**: View the optimization process, including:
   - **History of Study**: Shows how the objective value changed over different trials.
   - **History of Hyperparameters**: Displays the evolution of the four key hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) over the trials.
   - **Importance of Hyperparameters**: Provides insights into which hyperparameters had the most impact on the model's performance.

## 10. Future Improvements

Our model is great, but even Einstein had room for improvement. Here are some ideas:

1. **Try Other Algorithms**: Maybe a Gradient Boosting model or a Neural Network could be our next fortune-teller. XGBoost, anyone?

2. **Gather More Data**: The more, the merrier (and potentially more accurate). Maybe we could include data on company performance or local job markets?

3. **Deep Dive into High-Risk Groups**: Create separate models for different departments or job levels. The secret sauce for keeping engineers might be different from what works for sales teams.

4. **Explainable AI**: Make our model's decisions as clear as your boss's coffee mug. Look into LIME (Local Interpretable Model-agnostic Explanations) for individual prediction explanations.

5. **Real-time Predictions**: Turn this into a living, breathing attrition prediction system. Integrate it with HR systems for continuous monitoring.

6. **Feature Engineering**: Create new features that might be predictive. Maybe the ratio of salary to years of experience could be telling?

7. **Survival Analysis**: Instead of just predicting if an employee will leave, predict when they might leave. It's like giving an expiration date to employee tenure!

## 11. Contributing

We welcome contributors with open arms (and open-source licenses)! Here's how you can join the attrition-predicting party:

1. Fork the repo (it's like photocopying our top-secret playbook)
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a pull request (and maybe include a joke or two in your commit messages)

Remember: With great pull requests comes great responsibility!

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. It's like the terms and conditions, but you might actually read this one!

---

Now go forth and predict! May your employees be happy, your attrition be low, and your coffee be strong. 💼☕🚀

*
