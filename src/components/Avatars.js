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

function Avatars({ id, onClick, avatarIdClicked, containerAPP }) {

  const cApp = containerAPP;
  return (
    <div className="teampict">
      {cApp !== 2
        ? projAvats[id + 1].map((idA) => (
          <img

            onClick={() => {
              onClick(idA);
            }}
            index={idA}
            key={idA}
            src={avatars[idA]}
            alt={avatars[idA]}
            style={{
              cursor: "pointer",
              opacity:
                avatarIdClicked === idA || avatarIdClicked === 0
                  ? "1"
                  : "0.3",
            }}

          />
        ))
        : projAvats[id + 1].map((idA) => (
          <img
            index={idA}
            key={idA}
            src={avatars[idA]}
            alt={avatars[idA]}
            style={{
              opacity: "1",
            }}
            height="32px"
            width="32px"
          />
        ))}
    </div>
  );
}
export default Avatars;
