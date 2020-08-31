import React from "react";
import { historyTeamData } from "./data.js";

function TeamHist({ id, onClick, avatarIdClicked }) {
  return (
    <div className="teamHistory" key={id} onClick={onClick}>
      {Math.max(...historyTeamData[id][avatarIdClicked]) !== 0 ? (
        <>
          {historyTeamData[id][avatarIdClicked].map((data, index) => (
            <div
              className="th"
              key={index}
              style={{
                backgroundColor:
                  data === 0
                    ? "white"
                    : data.toString().charAt(0) === "p"
                    ? "lightgray"
                    : data > 1
                    ? "red"
                    : "lightblue",
              }}
            >
              <p
                className="thg"
                style={{
                  height:
                    data.toString().charAt(0) === "p"
                      ? parseFloat(data.slice(1)) * 100 + "%"
                      : data * 100 + "%",
                  backgroundColor: data > 1 ? "orange" : "rgb(0, 102, 128)",
                }}
              >
                _
              </p>
            </div>
          ))}
        </>
      ) : (
        <></>
      )}
    </div>
  );
}
export default TeamHist;
