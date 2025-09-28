# Retail Price Optimization - Help them to sell more

In today's competitive retail market, setting the right price for products is crucial. This project focuses on retail price optimization using machine learning techniques to predict customer satisfaction scores. This is a crucial part of developing dynamic pricing strategies, leading to increased sales and customer satisfaction.

**Problem statement:** Our task is to develop a model that predicts the optimal price for a product based on various factors. This prediction would enable us to make an informed decision when pricing a product, leading to maximized sales and customer satisfaction.

The dataset's diverse features like product details, order details, review details, pricing, competition, time, and customer details provide a comprehensive view for our price optimization task.

By analyzing this information, we aim to predict the optimal price for retail products. This would aid in making strategic pricing decisions, thereby optimizing retail prices effectively.

The sample dataset includes various details about each order, such as:

- Product details: ID, category, weight, dimensions, and more.
- Order details: Approved date, delivery date, estimated delivery date, and more.
- Review details: Score and comments.
- Pricing and competition details: Total price, freight price, unit price, competitor prices, and more.
- Time details: Month, year, weekdays, weekends, holidays.
- Customer details: ZIP code, order item ID.


## 🚀 Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest`: Ingests the data from the databas into the ZenML repository.
- `categorical_encoder`: Encodes the categorical features of the dataset.
- `feature_engineer`: Create new features from the existing features.
- `split`: Splits the dataset into train and eval splits.
- `train`: Trains the model on the training split.
- `evaluate`: Evaluates the model on the eval split.
- `decision`:
- `deploy`: Deploys the model to a BentoML endpoint.
