
---

# **Project Report: [Linear Regression Implementation by Python]**

---

**Course Name:** AI and Machine Learning
**Course Code:** SDM274



**Submitted By:**
*   **Name:** `Lyu Zixuan`
*   **Student ID:** `12210201`
*   **Email:** `[12210201@mail.sustech.edu.cn]`

**Date of Submission:** `[26, September, 2025]`

---

### **Abstract**

**[Write this last. Summarize the entire report in ~250 words.]**
This report documents the design, implementation, and evaluation of the `[Project Name]`. The project aims to `[state the primary objective]` by addressing `[the specific problem or need]`. The methodology involved `[briefly describe the main techniques, tools, or processes used, e.g., agile development, experimental research, user-centered design]`. Key outcomes include `[state the main results, e.g., a functional web application that performs X, experimental data proving Y, a designed prototype that improves Z]`. The findings demonstrate that `[state the main conclusion]`. This project serves as `[mention the significance or implication of your work]`.

**Keywords:** `Linear Regression`, `Gradient descent`, `Least square solution`
---



### **1. Introduction**

**1.1. Project Background & Motivation**
[Linear regression is a fundamental machine learning technique for predicting continuous outputs, such as housing prices or sales trends. While the least squares method provides exact solutions, it can be inefficient for large datasets. Gradient descent methods—SGD, BGD, and MBGD—offer scalable alternatives and are widely used in practice. At the same time, data normalization plays a key role in improving convergence and avoiding bias caused by different feature scales. This project is motivated by the need to evaluate how optimization strategies and normalization techniques influence the training and performance of linear regression models.]

**1.2. Summary of Your Project**
[In this project, I implemented a Linear Regression model in Python from scratch using Numpy. The model was designed to minimize Mean Squared Error (MSE) loss through three optimization strategies: Stochastic Gradient Descent (SGD), Batch Gradient Descent (BGD), and Mini-Batch Gradient Descent (MBGD). In addition, I integrated two data normalization methods—min-max normalization and mean normalization—to study their effects on training speed and stability. The project includes generating synthetic datasets, implementing the algorithms, visualizing convergence behaviors, and analyzing the results under different configurations.]

### **2. Problem Description and Project Objectives**
**2.1 Problem Description** 
[The problem addressed in this project is to design and implement a linear regression model capable of learning the relationship between an input variable and a continuous target variable. Instead of relying on closed-form solutions such as the least squares method, the focus is on implementing iterative optimization techniques—namely Stochastic Gradient Descent (SGD), Batch Gradient Descent (BGD), and Mini-Batch Gradient Descent (MBGD)—to minimize the Mean Squared Error (MSE) loss function.

Additionally, the project aims to investigate the role of data normalization in improving convergence behavior. Without normalization, features with different scales may slow down training or lead to unstable results. By comparing min-max normalization, mean normalization, and the unnormalized baseline, the project seeks to identify how preprocessing impacts performance and stability.]



**2.2Primary Objective:** 
[The specific aims of this project are:

To implement a Linear Regression class in Python using Numpy, supporting gradient descent–based training.

To ensure robustness and efficiency by incorporating three optimization strategies: SGD, BGD, and MBGD.

To integrate data normalization methods (min-max and mean normalization) into the training process.

To evaluate the effectiveness of different methods by analyzing convergence speed, stability, and accuracy using a synthetic dataset and visualizations with Matplotlib.]



### **3. Design & Methodology**

**3.1. System Architecture / Overall Design**
[The project follows a **modular design** to ensure clarity and reusability. The architecture can be summarized in four main stages:  

1. **Data Generation**  
   - A synthetic dataset is generated using the equation:  
     \[
     y = ax + b + \text{noise}
     \]  
     where noise follows a Gaussian distribution.  

2. **Preprocessing**  
   - Input features can be left raw, or normalized using either **min-max normalization** or **mean normalization**.  
   - Normalization is applied before model training to improve stability and convergence speed.  

3. **Model Training (LinearRegression Class)**  
   - The class encapsulates key functions:  
     - `fit()` — trains the model using **SGD**, **BGD**, or **MBGD**.  
     - `predict()` — computes predictions on new data.  
     - `loss()` — calculates Mean Squared Error (MSE).  

4. **Evaluation & Visualization**  
   - Loss curves are plotted to analyze convergence speed.  
   - Regression lines are visualized on training data to evaluate performance qualitatively. ]

**3.2. Implementation Details**
[Describe *how* you built the core components. This is a technical deep dive. Use pseudo-code, code snippets, class diagrams, or screenshots of the UI to illustrate your points. Focus on 2-3 key features or challenges.]


### **4. Testing & Results**

**4.1. Testing**
[Describe how you tested your project (e.g., Unit Testing, Integration Testing, User Acceptance Testing). List the types of test cases you executed.]

**4.2. Results and Analysis**
[Present the outcomes of your testing. Use tables, graphs, and charts to display data. Did you meet your requirements?]
*   **Table: Functional Test Results**
| Test Case ID | Description | Input | Expected Result | Actual Result | Status (Pass/Fail) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TC-01 | User Login | Valid Credentials | Access Granted | Access Granted | Pass |
| TC-02 | User Login | Invalid Password | Error Message | Error Message | Pass |
*   **Analysis:** [Explain what the results mean. "The results confirm that all functional requirements were met. The performance test showed an average response time of 1.5s, which is under our 2s target."]


### **5. Conclusion**


[Concisely summarize the project. Re-state the problem, your approach, and the key findings. affirm that you have met your objectives.]


[Reflect on the process. What was difficult? What did you learn technically and professionally (e.g., project management, teamwork)?]

