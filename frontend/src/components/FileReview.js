import React, { useEffect, useState } from "react";
import axios from "axios";
import LoadingComponent from "./LoadingComponent";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

export default function FileReview() {
  const [fileInfo, setFileInfo] = useState(null);
  const navigate = useNavigate();
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/get_extracted_files"
        );
        setFileInfo(response.data);
        toast.success("files retrieved");
      } catch (error) {
        console.error("Error fetching file information:", error);
      }
    };

    if (fileInfo === null) {
      fetchData();
    }
  }, []);

  return (
    <div>
      {fileInfo ? (
        <div className="w-full h-[90vh] flex justify-center items-center">
          <div className="w-[40%] h-[90%] shadow-xl border-2 rounded-2xl">
            <div className="flex justify-center ">
              <h1 className="p-2 m-2 text-xl uppercase text-center ">
                {" "}
                Number of files:{" "}
              </h1>
              <h1 className="p-2 m-2 text-xl font-semibold uppercase text-center ">
                {fileInfo.num_files}
              </h1>
            </div>

            <h3 className="p-2 text-xl font-semibold">File Names:</h3>

            <hr />
            <ul className=" overflow-y-scroll h-[60%] w-full ">
              {fileInfo.file_names.map((fileName, index) => (
                <li key={index} className=" text-lg text-center p-2">
                  {fileName}.txt <hr />
                </li>
              ))}
            </ul>
            <hr />
            <div className="w-[95%] flex justify-end p-2 m-2">
              <button
                onClick={() => navigate("/preprocessing")}
                type="submit"
                className="bg-blue-800 hover:bg-blue-900 text-white font-bold py-2 px-2 rounded-xl md:w-[40%] w-[100%] my-4 md:my-0"
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      ) : (
        <LoadingComponent />
      )}
    </div>
  );
}
