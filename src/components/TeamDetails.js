import React, { useState, useEffect } from "react";
import { teamData } from "./data.js";
import { teamviewsingle } from "./data.js";
import avatars1 from "../avatars1.png";
import avatars2 from "../avatars2.png";
import avatars3 from "../avatars3.png";
import avatars4 from "../avatars4.png";
import avatars5 from "../avatars5.png";
import avatars6 from "../avatars6.png";
import avatars7 from "../avatars7.png";
import avatars8 from "../avatars8.png";

function TeamDetails({ teamMateClicked, onClick }) {
  const [profileState, setProfileState] = useState(teamMateClicked);

  useEffect(() => {
    setProfileState(teamMateClicked);
  }, [teamMateClicked]);

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

  var d = new Date();
  var n = d.getHours();

  var pid = profileState;
  const id = 0;
  const sumvalues =
    teamviewsingle[0][pid][0] +
    teamviewsingle[1][pid][0] +
    teamviewsingle[2][pid][0];

  return (
    <div className="projects">
      <h1
        onClick={onClick}
        style={{
          color: "dimgray",
        }}
      >
        TEAM DETAILS VIEW
      </h1>

      <div className="projArea">
        <div className={`projcont`}>
          <div className="projName">
            <div
              className="fieldProjectName"
              onClick={onClick}
              style={{ cursor: "pointer" }}
            >
              {teamData[pid].persname}
            </div>
          </div>
          <div className="projDescription">
            {" "}
            <div className="projLeer" onClick={onClick}>
              {sumvalues > 0.01 ? (
                <>
                  <div
                    className="teamProgress"
                    style={{
                      height: sumvalues * 100 + "%",
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
                      backgroundColor: teamviewsingle[id][pid][2],
                      top: 95 - sumvalues * 100 + "%",
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
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont${teamData[pid].personWorkExperience > 0.9 ? "red" : ""}`}>
          <div className="projName">
            <div className="fieldProjectName">Personal data</div>
          </div>
          <div className="projDescription">
            <div className="projLeer">
              <div className="personalData"><span>ID nr. {pid}</span></div>
              <div className="personalData"><span>Phone</span></div>
              <div className="personalData"><span>Mail</span></div>
            </div>
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont`}>
          <div className="projName">
            <div className="fieldProjectName">Experience</div>
          </div>
          <div className="projDescription">
            <div className="projLeer">
              <div
                className="projTimeDiagram"
                style={{
                  width: teamData[pid].personWorkExperience * 100 + "%",
                }}
              ></div>
              <div
                className="dotLineVertical"
                style={{
                  left: teamData[pid].personWorkExperience * 100 + "%",
                  backgroundColor: "rgb(127 145 187)",
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont`}>
          <div className="projName">
            <div className="fieldProjectName">Keywords</div>
          </div>
          <div className="projDescription">
            <div className="projLeer" style={{ pointerevents: "none" }}>
              <div className="textDescriptionPD">
                <span>{teamData[pid].personKeywords}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="AvatarCarussell">
        <div className="AvatarCarussellSides" onClick={() => onClick(teamMateClicked = (teamMateClicked > 1 ? pid - 1 : 8))}> </div>
        <img

          index={pid}
          key={pid}
          src={avatars[pid]}
          alt={avatars[pid]}

        />
        <div className="AvatarCarussellSides" onClick={() => onClick(teamMateClicked = (teamMateClicked < 8 ? pid + 1 : 1))}></div>
      </div>
    </div >
  );
}
export default TeamDetails;
