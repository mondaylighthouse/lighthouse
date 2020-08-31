import React, { useState } from "react";
import { projects } from "./data.js";
import Avatars from "./Avatars";
import { teamData } from "./data.js";

function SetupNewProject() {
  const [whichAvatar, setwhichAvatar] = useState(0);
  const [stadium, setstadium] = useState(0);
  const newteam = [];

  console.log("newteam" + newteam[0]);
  console.log(stadium + " " + whichAvatar);
  const av = (id) => {
    console.log("id" + id);
    setwhichAvatar(id);
    setstadium(stadium + 1);
  };

  const changeStadium = () => {
    stadium === 3 && setstadium(4);
    setstadium(stadium + 1);
  };

  return (
    <div className="setup">
      {stadium === 0 && ( //logical conditional operator stadium = 0 && <div>....
        <div className="setupColumn">
          <label>Please enter project name (ID: {projects.length + 1}):</label>
          <input type="text" id="lname" name="lname"></input>
          <div className="setupColumn">
            <button onClick={changeStadium}>NEXT</button>
          </div>
        </div>
      )}
      <div className="setupColumn">
        {stadium > 0 && (
          <Avatars onClick={av} avatarIdClicked={whichAvatar} id={-1} />
        )}
        {stadium === 1 && (
          <div className="setupColumn">
            <label style={{ textAlign: "center" }}>Choose your team!</label>
          </div>
        )}
      </div>

      {whichAvatar && stadium > 1 && (
        <>
          <div className="setupColumn">
            <div className="likeInput">
              Add {teamData[whichAvatar].persname} to project!
            </div>
          </div>
          <div className="setupColumn">
            <label>The new team:</label>
            <Avatars onClick={av} avatarIdClicked={whichAvatar} id={0} />
          </div>
          <div className="setupColumn">
            <button>Submit</button>
          </div>
        </>
      )}
    </div>
  );
}
export default SetupNewProject;
