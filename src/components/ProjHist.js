import React from "react";
import { historyData } from "./data.js";

function ProjHist({ id, onClick }) {
  return (
    <div className="projectHistory" key={id} onClick={onClick}>
      {historyData[id].map((data, index) => (
        <div
          className="ph"
          key={index}
          style={{
            backgroundColor:
              data === 0
                ? "white"
                : data.toString().charAt(0) === "p"
                ? "lightgray"
                : data > 1
                ? "red"
                : "lightgreen",
          }}
        >
          <p
            className="phg"
            style={{
              height:
                data.toString().charAt(0) === "p"
                  ? parseFloat(data.slice(1)) * 100 + "%"
                  : data * 100 + "%",
              backgroundColor: data > 1 ? "orange" : "green",
            }}
          >
            _
          </p>
        </div>
      ))}
    </div>
  );
}
export default ProjHist;
