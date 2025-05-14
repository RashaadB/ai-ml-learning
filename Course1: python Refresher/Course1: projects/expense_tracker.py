{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4119297f",
   "metadata": {},
   "source": [
    "1.    Design and implement a personal expense tracker that enables users to manage their expenses\n",
    "2.    Allow users to categorize expenses and set monthly budgets\n",
    "3.    Implement file-handling functionality to save and load expense data\n",
    "4.    Create an interactive, menu-driven interface for ease of use"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d70da6a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datetime import datetime\n",
    "\n",
    "def get_expense_input():\n",
    "    while True:\n",
    "        date_input = input(\"Enter the date of the expense (YYYY-MM-DD): \")\n",
    "        try:\n",
    "            date = datetime.strptime(date_input, \"%Y-%m-%d\").date()\n",
    "            break\n",
    "        except ValueError:\n",
    "            print(\"Invalid date format. Please use YYYY-MM-DD.\")\n",
    "    \n",
    "    while True:\n",
    "        category = input(\"Enter the category of the expense (e.g., Food, Travel): \").strip()\n",
    "        if category:\n",
    "            break\n",
    "        print(\"Category cannot be empty.\")\n",
    "    \n",
    "    while True:\n",
    "        amount_input = input(\"Enter the amount spent: \")\n",
    "        try:\n",
    "            amount = float(amount_input)\n",
    "            if amount >= 0:\n",
    "                break\n",
    "            else:\n",
    "                print(\"Amount must be non-negative.\")\n",
    "        except ValueError:\n",
    "            print(\"Invalid amount. Please enter a number.\")\n",
    "    \n",
    "    description = input(\"Enter a brief description of the expense: \").strip()\n",
    "    \n",
    "    return {\n",
    "        \"date\": str(date),\n",
    "        \"category\": category,\n",
    "        \"amount\": amount,\n",
    "        \"description\": description\n",
    "    }\n",
    "\n",
    "def display_expenses(expenses):\n",
    "    if not expenses:\n",
    "        print(\"No expenses recorded.\")\n",
    "        return\n",
    "\n",
    "    print(\"\\nStored Expenses:\")\n",
    "    print(\"-\" * 60)\n",
    "    for expense in expenses:\n",
    "        print(f\"Date: {expense['date']} | \"\n",
    "              f\"Category: {expense['category']} | \"\n",
    "              f\"Amount: ${expense['amount']:.2f} | \"\n",
    "              f\"Description: {expense['description']}\")\n",
    "    print(\"-\" * 60)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6ae1d39",
   "metadata": {},
   "source": [
    "- datetime was imported to get a valid datetime.date from the input \n",
    "\n",
    "I used while loops so they keep running until they are interrupted by the break. \n",
    "\n",
    "- The first While loop prompts the user to put in a valid date in the YYYY-MM-DD format. \n",
    "- The second while loop prompts the user to put in a category as a string and it cannot be empty spaces. And .strip() removes leading and trailing spaces\n",
    "- the third while loop promts the user to put in a dollar amount as a float number. the amount must be greater or equal to 0. \n",
    "- the function then returns the date, category, amount, and description when its called. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66cf1b39",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Add a new expense\n"
     ]
    }
   ],
   "source": [
    "expenses = []\n",
    "\n",
    "while True:\n",
    "    print(\"\\nAdd a new expense\")\n",
    "    expense = get_expense_input()\n",
    "    expenses.append(expense)\n",
    "\n",
    "    another = input(\"Do you want to add another expense? (y/n): \").strip().lower()\n",
    "    if another != 'y':\n",
    "        break\n",
    "\n",
    "display_expenses(expenses)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
