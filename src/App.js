import React, { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import GeneralOverView from "./components/GeneralOverView";
import ProjectDetails from "./components/ProjectDetails.js";
import SetupPU from "./components/SetupPU.js";
import TeamDetails from "./components/TeamDetails";
import Predictionboard from "./components/Predictionboard.js";
//import useSound from "use-sound";
//import click from "./click.mp3";
import TeamOverView from "./components/TeamOverView";

function App() {
  const [whichAppContainer, setwhichAppContainer] = useState(0);
  const [whichProjectClicked, setwhichProjectClicked] = useState(0);
  var [whichTeamMateClicked, setwhichTeamMateClicked] = useState(0);
  //const [play] = useSound(click); //initiate the var for sound effect //Here clear hot to switch the sound of tried with state but conditional calling of hook

  const setup = (pid) => {
    setwhichTeamMateClicked(pid);
    pid > 0
      ? setwhichAppContainer(3)
      : whichAppContainer === 0
      ? setwhichAppContainer(1)
      : setwhichAppContainer(0); //vorher 0 für landing page
  };

  const setupnew = (i) => {
    console.log("get" + i);
    whichAppContainer === 0 ? setwhichAppContainer(i) : setwhichAppContainer(0); //vorher 0 für landing page
  };

  const switchToPD = (id, ToP) => {
    //ToP clears how to get id of the calling element
    //ToP clears how to get id of the calling element
    ToP === 1 ? setwhichAppContainer(2) : setwhichAppContainer(3);
    setwhichProjectClicked(id);
    setwhichTeamMateClicked(id);
  };

  return (
    <div className="App">
      <Header onClick={setupnew} />
      {
        {
          "0": (
            <GeneralOverView
              onClick={switchToPD}
              containerAPP={whichAppContainer} //important for switch between PDV <> PLV
            />
          ),
          "1": <SetupPU />,
          "2": (
            <ProjectDetails
              onClick={setup}
              containerAPP={whichAppContainer} //important for Avatar not-clickable in PDV
              projectClicked={whichProjectClicked}
            />
          ),
          "3": (
            <TeamDetails
              onClick={setup}
              teamMateClicked={whichTeamMateClicked}
            />
          ),
          "4": <Predictionboard />,
          "5": (
            <TeamOverView
              onClick={switchToPD}
              containerAPP={whichAppContainer}
            />
          ),
        }[whichAppContainer]
      }
    </div>
  );
}

export default App;
