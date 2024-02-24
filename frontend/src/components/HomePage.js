import React from "react";
import { FaArrowRight } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
export default function HomePage() {
  const navigate = useNavigate();
  return (
    <div className=" w-full h-[90vh]">
      <div className="flex w-full h-full flex-col px-11 justify-center gap-8">
        <div className="text-7xl ">
          Text Similarity Checker using Machine Learning
        </div>
        <div className=" w-[90%] text-justify font-inter">
          Text similarity checker is an advanced tool that assists in the
          classification and categorization of various documents. Users are able
          to effectively analyze and organize their textual content, ensuring
          that information is properly sorted and easily accessible. This
          solution enhances document management processes, enabling users to
          efficiently handle large volumes of data and extract valuable
          insights.
        </div>
        <div>
          <button onClick={() => {navigate("/input")}} className=" px-4 py-2 border-2 hover:rounded-xl rounded-md duration-300 border-[#c3e6ff] hover:border-[#8ccdff] text-xl bg-[#c3e6ff]  hover:bg-[#8ccdff] flex items-center gap-2 uppercase">
            Upload files <FaArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
