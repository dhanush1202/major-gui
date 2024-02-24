import React, { useEffect, useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import { useNavigate } from "react-router-dom";

export default function Preprocessing() {
  const [fileInfo, setFileInfo] = useState(null);
  const [simvisible, setsimvisible] = useState(false);
  const algorithms = ["KNN", "Decision Tree", "Logistic Regression", "MLP"];
  const dt_methods = ["Without_Descretization", "With_Descretization", "Gaussian"];
  // const knn_methos = ["Euclidean", "Cosine"]
  // const lr_methods = ["Standard Scaling", "Constant Threshold", "Stratified-K_fold"]
  const algo_sim = {
    "Decision Tree": ["Without_Descretization", "With_Descretization", "Gaussian"],
    "KNN": ['Euclidean', 'Cosine'],
    "MLP": ['2 Layers','3 Layers', '4 Layers'],
    "Logistic Regression": ['Constant Threshold', 'Stratified K-Fold', 'Gaussian', 'Variable Threshold']
  }
  const [selectedalgo, setselectedalgo] = useState("");
  const [selectedsim, setselectedsim] = useState("");
  const navigate = useNavigate();
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response3 = await axios.get(
          "http://localhost:5000/generate_representation"
        );
        setFileInfo(response3.data);
        toast.success("Information Gain calculated");
        console.log(response3.data);
      } catch (error) {
        console.error("Error fetching file information:", error);
      }
    };

    // Only make the API calls if fileInfo is null
    if (!fileInfo) {
      fetchData();
    }
  }, [fileInfo]); // Adding fileInfo to the dependency array to trigger the effect when it changes

  
  const algor = (item) => {
    setselectedalgo(item);
    setsimvisible(true);
  }

  const simsubmitted = async (item) => {
    setselectedsim(item);
    const data = {
      selectedalgo: selectedalgo,
      item: item
    };
    try {
      const response = await axios.post("http://localhost:5000/runalgo", data);
      console.log(response);
      navigate("/dashboard");
    } catch (err) {
      console.log(err);
    }
}

  return (
    <div>
      {fileInfo ? (
        <div className="flex justify-center items-center w-screen h-[90vh]">
          <div className=" w-[80%] rounded-xl flex-col border-2 h-[90%] justify-center gap-6 flex items-center  shadow-xl">
            <div className="flex flex-col justify-center items-center gap-4">
              <div className=" text-4xl font-inter ">
                Select the algorithm for classification
              </div>
              <div className="w-[80%] text-justify">
                Lorem ipsum, dolor sit amet consectetur adipisicing elit.
                Repellendus, in provident eum tempore reiciendis perferendis
                dolores corporis. Saepe facilis corrupti, unde commodi,
                voluptatibus ex nisi quos, porro iure incidunt eaque?
              </div>
            </div>

            <div className="flex w-full justify-evenly">
              {" "}
              {Object.keys(algo_sim).map((item, key) => (
                <div
                  key={key} onClick={() => algor(item)}
                  className=" px-4 py-2 border-[1px] border-blue-400 hover:border-blue-300 font-inter bg-blue-400 hover:bg-blue-300 cursor-pointer rounded-xl duration-200 text-2xl "
                >
                  {item}
                </div>
              ))}
            </div>
            {simvisible && (
              <>
                <div className=" text-2xl font-inter text-left w-[80%]">
                  Method:
                </div>
                <div className="flex w-full justify-evenly">
                  {" "}
                  {Object.values(algo_sim[selectedalgo]).map((item, key) => (
                    <div onClick={() => simsubmitted(item)}
                      key={key}
                      className=" px-4 py-2 border-[1px] border-blue-400 hover:border-blue-300 font-inter bg-blue-400 hover:bg-blue-300 cursor-pointer rounded-xl duration-200 text-2xl "
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      ) : (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(255, 255, 255, 0.8)",
            zIndex: 9999,
          }}
        >
          Loading
        </div>
      )}
    </div>
  );
}
