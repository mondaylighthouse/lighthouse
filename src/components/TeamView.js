import React from "react";
import { teamviewsingle } from "./data.js";

var d = new Date();
var n = d.getHours();

function TeamView({ id, onClick, avatarIdClicked }) {
  return (
    <div className="projLeer" onClick={onClick}>
      {teamviewsingle[id][avatarIdClicked][0] > 0.01 ? (
        <>
          <div
            className="teamProgress"
            style={{
              height: teamviewsingle[id][avatarIdClicked][0] * 100 + "%",
            }}
          ></div>
          <div
            className="projBusy"
            style={{
              width: n * 4.1 + 0.5 + "%",
              opacity: 0.4,
            }}
          ></div>
          <div
            className="dotDay"
            style={{
              backgroundColor: teamviewsingle[id][avatarIdClicked][2],
              top: 95 - teamviewsingle[id][avatarIdClicked][0] * 100 + "%",
              left: n * 4.1 - 1 + "%",
            }}
          ></div>
          <div
            className="dotLineVertical"
            style={{
              left: n * 4.1 + 1 + "%",
            }}
          ></div>
        </>
      ) : (
          <></>
        )}
    </div>
  );
}

export default TeamView;
