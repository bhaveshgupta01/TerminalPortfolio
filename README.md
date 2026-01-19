
# 🖥️ Terminal Portfolio v2

> A retro-futuristic personal portfolio blending Command Line aesthetics with modern, responsive UX. 
> Built with **React**, **Tailwind CSS**, and **Vite**.

<img width="1470" height="831" alt="Project Screenshot" src="https://github.com/user-attachments/assets/db6255e3-f927-4c70-87b9-0235462eb84f" />

## ✨ Key Features

- **🎨 Hybrid Design:** Combines a developer-centric "Terminal" theme with a visual "Netflix-style" gallery for creative work.
- **📱 Fully Responsive:** Optimized for all devices using Tailwind's mobile-first utility classes.
- **🧩 Config-Driven Content:** All project data, skills, and experience are managed via a single `portfolio.js` file, making updates instant.
- **⚡ High Performance:** Built on Vite for lightning-fast HMR and optimized production builds.
- **🖼️ Custom Components:**

  - **Design Carousel:** Horizontal snap-scrolling for design assets.
  - **Project Modals:** Detailed pop-ups with "Click-to-View" functionality.
  - **Lightbox:** Full-screen image viewer for design portfolios.

## 🛠️ Tech Stack

- **Frontend:** React.js, Vite
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Routing:** React Router DOM
- **Deployment:** Vercel (Recommended)

## 🚀 Getting Started

### Prerequisites
Make sure you have Node.js installed.

### Installation

1. **Clone the repo**
```bash
git clone [https://github.com/bhaveshgupta01/TerminalPortfolio.git](https://github.com/bhaveshgupta01/TerminalPortfolio.git)
cd TerminalPortfolio
```

2. **Install dependencies**
```bash
npm install

```


3. **Start the development server**
```bash
npm run dev

```



## 📂 Project Structure

```
src/
├── components/    # Reusable UI components (Cards, Modals)
├── data/          # CENTRAL DATA SOURCE (portfolio.js, designs.js)
├── pages/         # Page Views (Home, Design, Publications)
├── assets/        # Static images
└── App.jsx        # Main Router Logic

```

## 🔮 Future Roadmap (Agentic AI)

The next phase of this project involves integrating a **Python (FastAPI)** backend to enable **Agentic AI** capabilities:

* [ ] **LangGraph Orchestration:** An AI agent that navigates the website based on user queries.
* [ ] **RAG (Retrieval Augmented Generation):** Chatbot that answers questions about my resume using vector search.
* [ ] **Voice Command Interface:** Navigate the terminal using voice.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for the "Agentic AI" phase, feel free to open an issue.

## 📄 License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
