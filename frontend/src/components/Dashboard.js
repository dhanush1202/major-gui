import React, { useEffect, useState } from "react";
import axios from "axios";

export default function Dashboard() {
  const samplejson = {
    col1: ["col1 row1", "col1 row2", "col1 row3"],
    col2: ["col2 row1", "col2 row2", "col2 row3"],
    col3: ["col3 row1", "col3 row2", "col3 row3"],
  };
  const data = {
    item: "",
    selectedalgo: "",
    acc: 0,
    bal_acc: 0,
    f1_measure:0,
    recall:0,
    precison:0
  };
  const [resdata, setresdata] = useState({});

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await axios.get("http://localhost:5000/get_results");
        console.log(response.data.data)
        setresdata(response.data.data)

        // Further processing of data as needed
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    }
    fetchData();
  }, []);
  const transposedData = [];
  for (let i = 0; i < samplejson.col1.length; i++) {
    const row = [];
    for (const prop in samplejson) {
      row.push(samplejson[prop][i]);
    }
    transposedData.push(row);
  }
  return (
    <div className="flex justify-center items-center w-full h-[90vh]">
      <div className="w-full h-[85vh] flex justify-evenly ">
        <div className="w-[70%] h-full  flex flex-col items-center gap-4 ">
          <div className="w-full h-[30%] border-2 rounded-xl shadow-xl p-4 ">
            {" "}
            <div className="text-lg uppercase font-semibold">results:</div>
            <table className=" w-full h-[70%]">
              <tr>
                <td className="border-2 w-[33%] text-center">accuracy: {resdata.acc}</td>
                <td className="border-2 w-[33%] text-center">
                  {" "}
                  balanced accuracy: {resdata.bal_acc}
                </td>
                <td className="border-2 w-[33%] text-center"> f-score: </td>
              </tr>
              <tr>
                <td className="border-2 w-[33%] text-center">sdkbs:</td>
                <td className="border-2 w-[33%] text-center"> mhdgus:</td>
                <td className="border-2 w-[33%] text-center">kjhduysd:</td>
              </tr>
            </table>
          </div>
          <div className="w-full h-[60%] border-2 rounded-xl shadow-xl p-4 ">
            <div className="text-lg uppercase font-semibold">
              classes and their corresponding files
            </div>
            <div className="w-full h-[90%] overflow-y-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    {Object.keys(samplejson).map((item, key) => (
                      <th className="border-2 text-center" key={key}>
                        {item}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {transposedData.map((row, index) => (
                    <tr key={index}>
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex} className="border-2 text-center">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="w-[25%] border-2 rounded-xl shadow-xl h-fit p-4 ">
          <div>
            <span className=" font-semibold">Algorithm: </span>{" "}
            <span>{resdata.selectedalgo}</span>
          </div>
          <div>
            <span className=" font-semibold">Similarity measure: </span>{" "}
            <span>{resdata.item}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
