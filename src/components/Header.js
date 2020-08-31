import React from "react";
import logo from "../lighthouseLogo.png";
import team from "../team.png";
import chart from "../chart.png";

function Header({ onClick }) {
  return (
    <header className="App-header">
      <div className="imgCont" onClick={() => onClick(1)}>
        <img src={logo} className="App-logo" alt="logo" />
      </div>
      <div className="imgCont" onClick={() => onClick(5)}>
        <img src={team} className="App-logo" alt="team" />
      </div>
      <div className="imgCont" onClick={() => onClick(4)}>
        <img src={chart} className="App-logo" alt="chart" />
      </div>
    </header>
  );
}
export default Header;
