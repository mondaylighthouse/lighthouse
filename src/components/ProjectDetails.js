import React, { useState } from "react";
import Avatars from "./Avatars.js";
import ProjDiagram from "./ProjDiagram.js";
import { projects } from "./data.js"


function ProjectDetails({ projectClicked, containerAPP, onClick }) {
  const [whichAvatar] = useState(0);
  const pid = projectClicked;
  return (
    <div className="projects">
      <h1 onClick={onClick}
        style={{
          color: "dimgray",
        }}
      >
        PROJECT DETAILS VIEW
      </h1>

      <div className="projArea">
        <Avatars
          id={pid}
          avatarIdClicked={whichAvatar}
          containerAPP={containerAPP}
        />
        <div
          className={`projcont${projects[pid].rate === "busy" ? "red" : ""}`}
        >
          <div className="projName">
            <div
              className="fieldProjectName"
              onClick={onClick}
              style={{ cursor: "pointer" }}
            >
              {projects[pid].projname}
            </div>
          </div>
          <div className="projDescription">
            <ProjDiagram id={pid} />
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont${projects[pid].description ? "" : "red"}`}>
          <div className="projName">
            <div className="fieldProjectName">Description</div>
          </div>
          <div className="projDescription">
            <div className="projLeer" style={{ pointerevents: "none" }}>
              <div className="textDescriptionPD">
                <span>{projects[pid].description}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont${projects[pid].costs > 0.9 ? "red" : ""}`}>
          <div className="projName">
            <div className="fieldProjectName">Costs</div>
          </div>
          <div className="projDescription">
            <div className="projLeer">
              <div
                className="projCostDiagram"
                style={{
                  width: projects[pid].costs * 100 + "%",
                }}
              ></div>
              <div
                className="dotLineVertical"
                style={{
                  left: projects[pid].costs * 100 + "%",
                  backgroundColor: "rgb(182 214 188)",
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>
      <div className="projArea">
        <div className={`projcont${projects[pid].time > 0.9 ? "red" : ""}`}>
          <div className="projName">
            <div className="fieldProjectName">Time</div>
          </div>
          <div className="projDescription">
            <div className="projLeer">
              <div
                className="projTimeDiagram"
                style={{
                  width: projects[pid].time * 100 + "%",
                }}
              ></div>
              <div
                className="dotLineVertical"
                style={{
                  left: projects[pid].time * 100 + "%",
                  backgroundColor: "rgb(127 145 187)",
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default ProjectDetails;
