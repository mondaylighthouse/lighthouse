import React, { useState } from "react";
import SetupNewProject from "./SetupNewProject";

function SetupPU() {
  const [whichContainer, setwhichContainer] = useState(0);
  const snp = () => {
    setwhichContainer(1);
  };

  return (
    <div className="setupContainer">
      {
        {
          "0": (
            <>
              <div className="setup">
                <div className="setupTile">
                  <div className="likeInput" onClick={snp}>
                    <span className="setupButtons">Setup new project</span>
                  </div>
                </div>
                <div className="setupTile">
                  <div className="likeInput">
                    <span className="setupButtons">Change Teams</span>
                  </div>
                </div>
                <div className="setupTile">
                  <div className="likeInput">
                    <span className="setupButtons">Sound OFF</span>
                  </div>
                </div>
              </div>
            </>
          ),
          "1": (
            <>
              <SetupNewProject />
            </>
          ),
        }[whichContainer]
      }
    </div>
  );
}
export default SetupPU;
