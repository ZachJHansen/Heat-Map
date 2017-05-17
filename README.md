# Heat Map Documentation

This is a prototype web GUI for TTU’s HPCC. A line by line description for the main page is given below (still in progress). To run this application, download stable.html as a complete webpage and open in Chrome. (I know it works for Chrome, but I can’t speak for any other browsers). The buttons are stub functions. They will try to redirect you to your localhost:8888, where they expect an index.php containing links to other pages (like Graphs or Help). All this is still in-progress, so I would advise just not clicking the buttons until I can fix them.  

The Graphs.html code uses RGraph (https://github.com/seeslab/rgraph).


Start head.

Script Tags
Script tag one: imports jQuery 
Script tag two: imports a JavaScript graphics library

Style
Canvas: Main window the racks are drawn in
Pre: Standard output (do I even need this?)
Svg: The hover box will eventually be drawn in an svg viewbox superimposed on the canvas
H1: Title displayed above canvas
Img: Texas Tech logo in lower left corner

End head.

Start body

Title: “Heat Map of HPCC”
Canvas: specifications for canvas
Pre: specifications for output

Button One: Opens Help page
Button Two: Opens control panel page
Button Three: Opens Graphs page
Button Four: (Experimental) Requests Redfish payload from http://localhost:8080/redfish/v1/

Div element: “yellowBox” (change that name) Area to display information about selected machine

Svg element: Hoverbox. Still in the works

Begin script

Global variables with very self-explanatory names. 
Payload is a list of JSON objects (kinda) that each represent a server (also referred to as machines)

getJSON request that pulls json objects from an outside source and adds them to payload

Begin window onload function (called when the window loads)

Hoverbox stuff, don’t worry about for now
Lots of variables. Eventually the …Zone variables should be replaced with the criticalUpper, fatalUpper etc. variables from the Redfish payloads (check completeMetrics.json for examples)

Uncomment the data duplication section to fill the datacenter with machines displaying random greenZone metrics

Draw Row draws empty racks filling the canvas

Every 1000 ms, masterUpdate function is called and does the following
•	Every element in payload is parsed to isolate the name, temp, etc. of the machine
•	(Remember, each element in payload represents a server)
