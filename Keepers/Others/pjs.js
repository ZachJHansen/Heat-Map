// Waitfor.js example (PhantomJS example files)

"use strict";
function waitFor(testFx, onReady, timeOutMillis) {
    var maxtimeOutMillis = timeOutMillis ? timeOutMillis : 10000, //< Default Max Timout is 10s
        start = new Date().getTime(),
        condition = false,
        interval = setInterval(function() {
            if ( (new Date().getTime() - start < maxtimeOutMillis) && !condition ) {
                // If not time-out yet and condition not yet fulfilled
                condition = (typeof(testFx) === "string" ? eval(testFx) : testFx()); //< defensive code
            } else {
                if(!condition) {
                    // If condition still not fulfilled (timeout but condition is 'false')
                    console.log("'waitFor()' timeout");
                    phantom.exit(1);
                } else {
                    // Condition fulfilled (timeout and/or condition is 'true')
                    console.log("'waitFor()' finished in " + (new Date().getTime() - start) + "ms.");
                    typeof(onReady) === "string" ? eval(onReady) : onReady(); //< Do what it's supposed to do once the condition is fulfilled
                    clearInterval(interval); //< Stop this interval
                }
            }
        }, 250); //< repeat check every 250ms
};

var page = require('webpage').create();
var system = require('system');
var args = system.args;

// Default viewport
page.viewportSize = {
    width: 1920,
    height: 1080
};

if (args.length < 2 || args.length > 4){
    console.log("Please enter command: phantomjs pjs.js [url to render] [viewport width] [viewport height]");
    console.log("Viewport width/height are optional");
    phantom.exit(1);
} else {
    if (args.length == 4){
        page.viewportSize.width = args[2];
        page.viewportSize.height = args[3]; 
    }
    var url_partial = args[1];
    var url = "http://localhost:8888/" + url_partial;
    page.open(url, function (status) {
        // Check for page load success
        if (status !== "success") {
            console.log("Unable to access page");
	    phantom.exit(1);
        } else {
            // Wait for title to change (signals page is finished loading)
            waitFor(function() {
                var title = page.evaluate(function() {
                    return document.title;
                });
                if (title == "Loaded"){
                    return true;
                } else return false;
            }, function() {
                console.log("Content Loaded");
                if ((url_partial == 'Digital_Signage/QStat.html' || url_partial == 'Digital_Signage/PDU.html') || url_partial == 'Digital_Signage/storageAccessed.html'){
                    page.render('digitalSignage.png');
                } else {
                    console.log("Unknown url");
                }
                phantom.exit();
            });
        }
    });
}


