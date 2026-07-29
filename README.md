# DecoBot - Rule-Based Python Chatbot

A lightweight, rule-based conversational agent built in Python for **DecodeLabs**. **DecoBot** matches user inputs against a dictionary-backed knowledge base to provide instant automated responses.

---

## Features

* **Instant Dictionary Lookup:** Uses Python dictionaries for $O(1)$ constant-time response retrieval.
* **Input Preprocessing:** Normalizes raw input by lowercasing and stripping trailing whitespace to maximize match accuracy.
* **Graceful Fallbacks:** Handles unrecognized prompts with a polite default response.
* **Interactive Loop:** Continually accepts user input until explicitly terminated with the `exit` command.

---

## Project Structure

```text
.
├── main.py          # Core chatbot logic and interactive execution loop
└── README.md        # Project documentation
