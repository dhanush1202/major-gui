import React from "react";
import { useState } from "react";
import { Upload } from "keep-react";
import { Button } from "keep-react";
import { useForm } from "react-hook-form";
import axios from "axios";
import LoadingComponent from "./LoadingComponent";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
export default function InputPage() {
  const { control, handleSubmit, setValue } = useForm();
  const navigate = useNavigate();
  const [fileName, setFileName] = useState("");
  const [err, seterr] = useState(false);
  const [visibleload, setvisibleload] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      setValue("file", file);
    }
  };
  const validateFile = (data) => {
    if (data.file === undefined) {
      return true;
    }
    if (data.file.type === "application/x-zip-compressed") {
      return false;
    }
    return true;
  };
  const onFileSubmit = async (data) => {
    if (validateFile(data)) {
      seterr(true);
      return;
    }
    seterr(false);

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append("file", data.file);
    setvisibleload(true)
    try {
      // Use Axios to send the file to the Flask endpoint
     
      const response = await axios.post(
        "http://localhost:5000/upload_zip",
        formData
      );
      toast.success("unzipped")

      // Handle the response as needed
      console.log(response.data);
      navigate("/files_review");
    setvisibleload(false);

    } catch (error) {
      console.error("Error uploading the file:", error);
    }
    
  };

  return (
    <div>
    {visibleload && <LoadingComponent />}
    <div className="w-full h-[90vh] flex justify-center items-center">
     
      <div className="w-[40%] h-[90%] shadow-xl border-2 rounded-2xl">
        <h1 className="p-2 m-2 text-xl font-semibold uppercase text-center ">
          Upload the zip file here
        </h1>
        <form onSubmit={handleSubmit(onFileSubmit)}>
          <div className="">
            <Upload
              className="p-4 my-4"
              required
              onFileChange={(file) => {
                handleFileChange(file);
              }}
              file={fileName}
              fileType="Files accepted: Zip"
              title="Click choose file to upload file"
            />
          </div>
          {err && (
            <p className="text-center text-red-500 ">
              *File upload error. Please upload the file or check the file
              type.*
            </p>
          )}
          <div className="w-[95%] flex justify-end p-2 m-2">
            <button
              type="submit"
              className="bg-blue-800 hover:bg-blue-900 text-white font-bold py-2 px-2 rounded-xl md:w-[40%] w-[100%] my-4 md:my-0"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
    </div>
  );
}
