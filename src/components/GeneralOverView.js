import React, { useState } from "react";
import Avatars from "./Avatars.js";
import ProjDiagram from "./ProjDiagram.js";
import ProjHist from "./ProjHist.js";
import TeamHist from "./TeamHist.js";
import TeamView from "./TeamView";
import { projects } from "./data.js";
import { teamData } from "./data.js";

function GeneralOverView({ onClick, containerAPP }) {
  const [whichContainer, setwhichContainer] = useState(0);
  const [mateName, setmateName] = useState(null);
  const [whichAvatar, setwhichAvatar] = useState(0);

  const landingView = () => {
    setwhichContainer(0);
    setmateName(null);
    setwhichAvatar(0);
  };
  const pd = () => {
    setwhichContainer(1);
  };

  const tv = () => {
    setwhichContainer(3);
  };
  const th = () => {
    setwhichContainer(2);
  };

  const av = (id) => {
    // that is for switch view when clicking on Avatars TLV, HTV, PLV
    setwhichAvatar(id);
    setmateName(teamData[id].persname);
    id === whichAvatar
      ? landingView()
      : whichContainer === 3
      ? setwhichContainer(3)
      : setwhichContainer(2);
  };

  return (
    <div className="projects">
      {projects.length > 0 ? (
        <h1
          onClick={landingView}
          style={{
            cursor: "pointer",
            color: whichContainer < 2 ? "dimgray" : "midnightblue",
          }}
        >
          {
            {
              "0": "PROJECTS LIVE VIEW",
              "1": "PROJECTS HISTORY VIEW",
              "2": "TEAM LIVE VIEW",
              "3": "TEAM HISTORY VIEW",
            }[whichContainer]
          }
        </h1>
      ) : (
        <>
          <h1>YOU DON'T HAVE ANY PROJECTS!</h1>{" "}
          <h2>CLICK ON THE LOGO TO SETUP NEW PROJECTS</h2>{" "}
        </>
      )}
      {projects.map((data, indexProjects) => (
        <div className="projArea" key={indexProjects}>
          <Avatars
            containerApp={containerAPP} //important for Avatar not-clickable in PDV
            onClick={av}
            id={indexProjects}
            avatarIdClicked={whichAvatar}
          />

          <div className={`projcont${data.rate === "busy" ? "red" : ""}`}>
            <div className="projName">
              <div
                className="fieldProjectName"
                style={{ cursor: "pointer" }}
                onClick={() => onClick(indexProjects, 1)}
              >
                {data.projname}
              </div>
              <div
                className="fieldMateName"
                style={{ cursor: "pointer" }}
                onClick={() => onClick(whichAvatar, 2)} // only send values in correct order
              >
                {mateName}
              </div>
            </div>
            <div className="projDescription" key={indexProjects}>
              {
                {
                  "0": (
                    <ProjDiagram
                      onClick={pd}
                      id={indexProjects}
                      container={containerAPP}
                    />
                  ),
                  "1": <ProjHist onClick={landingView} id={indexProjects} />,
                  "2": (
                    <TeamView
                      onClick={tv}
                      id={indexProjects}
                      avatarIdClicked={whichAvatar}
                    />
                  ),
                  "3": (
                    <TeamHist
                      onClick={th}
                      id={indexProjects}
                      avatarIdClicked={whichAvatar}
                    />
                  ),
                }[whichContainer]
              }
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
export default GeneralOverView;
