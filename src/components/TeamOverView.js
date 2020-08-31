import React from "react";
import avatars1 from "../avatars1.png";
import avatars2 from "../avatars2.png";
import avatars3 from "../avatars3.png";
import avatars4 from "../avatars4.png";
import avatars5 from "../avatars5.png";
import avatars6 from "../avatars6.png";
import avatars7 from "../avatars7.png";
import avatars8 from "../avatars8.png";
import { projAvats } from "./data.js";
import { teamData } from "./data.js";

const avatars = {
  1: avatars1,
  2: avatars2,
  3: avatars3,
  4: avatars4,
  5: avatars5,
  6: avatars6,
  7: avatars7,
  8: avatars8,
};

function TeamOverView({ onClick, containerAPP }) {
  return (
    <div className="projects">
      <h1
        onClick={onClick}
        style={{
          cursor: "pointer",
          color: "dimgray",
        }}
      >
        "TEAM OVERVIEW"
      </h1>
      <div className="teamArea">
        {projAvats[0].map((indexTeam) => (
          <div className="projArea" key={indexTeam}>
            <img
              index={indexTeam}
              key={indexTeam}
              src={avatars[indexTeam]}
              alt={avatars[indexTeam]}
              style={{
                opacity: "1",
              }}
              height="48px"
              width="48px"
            />
            <div className="teamcont">
              <div
                className="projProgressTeam"
                style={{
                  minWidth: "80%",
                  height: teamData[indexTeam].personWorkExperience * 100 + "%",
                  backgroundColor: "lightsalmon",
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default TeamOverView;
