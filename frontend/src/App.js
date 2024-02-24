import React from "react";
import { Route, Routes } from "react-router-dom";
import { NavbarComponent } from "./components/Navbar";
import HomePage from "./components/HomePage";
import InputPage from "./components/InputPage";
import FileReview from "./components/FileReview";
import Preprocessing from "./components/Preprocessing";
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Dashboard from "./components/Dashboard";
export default function App() {
  return (
    <div className="w-[100vw] min-h-screen">
      <div className="">
        <NavbarComponent />
      </div>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="input" element={<InputPage />} />
        <Route path="/files_review" element={<FileReview />} />
        <Route path="/preprocessing" element={<Preprocessing />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
      <ToastContainer/>
    </div>
  );
}
