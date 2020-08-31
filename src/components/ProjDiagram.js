import React, { useState, useEffect, useRef } from "react";
import { projects } from "./data.js";
import { projDayData } from "./data.js";

var progressNr = 0;
function ProjDiagram({ id, onClick, container }) {
  var date = new Date(); //aktuelles Datum ermitteln
  var d = date.getDate(); //aktuellen Tag ermitteln z.B. 10. am 10.August
  var l = projDayData[id].length; //Projektlaenge in Tagen, Daten sind in data.js hinterlegt

  function useInterval(callback, delay) {
    //funktion zur ermittlung einer zuffalsnummer, nur fuer die DEMO
    const savedCallback = useRef();

    // Remember the latest callback.
    useEffect(() => {
      savedCallback.current = callback;
    }, [callback]);

    // Set up the interval.
    useEffect(() => {
      function tick() {
        savedCallback.current();
      }
      if (delay !== null) {
        let id = setInterval(tick, delay);
        return () => clearInterval(id);
      }
    }, [delay]);
  }

  let [ranNr, setranNr] = useState(0); //festlelegen der Zufallsnummer nur fuer die DEMO

  useInterval(() => {
    //Intervall fuer die aktualisierung der Anzeige nur fuer die DEMO
    setranNr(Math.random());
  }, 2000);

  progressNr < 100 ? (progressNr = progressNr + ranNr) : (progressNr = 0); //Check der Zufallsnummer nur duer die DEMO

  return (
    <div className="projLeer" onClick={onClick}>
      <div
        className="projProgress"
        style={{
          height: ranNr * 100 + "%",
        }}
      ></div>
      <div
        className="projBusy"
        style={{
          width: (d / l) * 100 + "%", //width: progressNr + "%", mit Zufallsnummer fuer die DEMO
          opacity: 0.4,
        }}
      ></div>
      <div
        className="dotDay" //platzieren den Statuspunkt
        style={{
          backgroundColor: projects[id].status,
          top: 95 - ranNr * 100 + "%", //top: 95 - ranNr * 100 + "%", mit Zufallsnummer fuer die DEMO
          left: (d / l) * 100 - 2 + "%", //left: progressNr - 2 + "%", mit Zufallsnummer fuer die DEMO
        }}
      ></div>

      {container !== 0 ? (
        projDayData[id].map((data, index) => (
          <div
            key={index}
            className="dotPrediction"
            style={{
              backgroundColor: index < d ? projects[id].status : "gray",
              top: 95 - data * 100 + "%",
              left: index * 4.1 - 1 + "%",
            }}
          ></div>
        ))
      ) : (
        <></>
      )}
      <div
        className="dotLineVertical"
        style={{
          left: (d / l) * 100 + "%", //left: progressNr + "%",  mit Zufallsnummer fuer die DEMO
        }}
      ></div>
    </div>
  );
}
export default ProjDiagram;
