/*
 * Complex Format EXPORT MACRO Plus Projection
 * By Olivier Burri @ EPFL - SV - PTECH - BIOP
 * Given a folder, extracts all series inside all multi-file files with given extension, max Projection 
 * and save in given output folder

 * Modified by Jana Tuennermann
 * Last edit: 29.09.2022
 */


 ////////////////////// SET PARAMETERS //////////////////////
 ////////////////////////////////////////////////////////////
 
 
// Set the extension you would like this macro to work with.
// Do not add a . at the beggining
extension = "nd";  //eg "lif", "vsi", etc...


// Set to true if you want series to be max-projected, if false series is resaved as stack tif
as_maxProjection = false; // set to either true or false
as_meanProjection = true

 //////////////////// END SET PARAMETERS ////////////////////
 ////////////////////////////////////////////////////////////
 




// Beggining of macro. You should now have anything to edit after this line. 

dir = getDirectory("Select a directory containing one or several ."+extension+" files.");

files = getFileList(dir);

outputdir = getDirectory("Select an output directory into which extracted data is saved");

setBatchMode(true);
k=0;
n=0;

run("Bio-Formats Macro Extensions");
for(f=0; f<files.length; f++) {
	if(endsWith(files[f], "."+extension)) {
		k++;
		id = dir+files[f];
		Ext.setId(id);
		Ext.getSeriesCount(seriesCount);
		print(seriesCount+" series in "+id);
		n+=seriesCount;
		for (i=0; i<seriesCount; i++) {
			run("Bio-Formats Importer", "open=["+id+"] color_mode=Default view=Hyperstack stack_order=XYCZT series_"+(i+1));
			fullName	= getTitle();
			//if (seriesCount>1) {
			//	fileName 	= substring(fullName, 0, lastIndexOf(fullName, " - ")-10);
			//} else {
			//	fileName 	= substring(fullName, 0, lastIndexOf(fullName, "-")+18);
			//}
			fileName 	= substring(fullName, 0, lastIndexOf(fullName, "WithSMB")+7);
			
			getDimensions(x,y,c,z,t);
			Stack.setXUnit("px");
			run("Properties...", "pixel_width=1 pixel_height=1 voxel_depth=1 frame=1");
			if(as_maxProjection) {
				run("Z Project...", "projection=[Max Intensity] all");
				saveAs("tiff", outputdir+File.separator+fileName+"_s"+(i+1)+"_MAX.tiff");
			} else if(as_meanProjection) {
				run("Z Project...", "projection=[Average Intensity] all");
				saveAs("tiff", outputdir+File.separator+fileName+"_s"+(i+1)+"_MEAN.tiff");
				
			} else {
				saveAs("tiff", outputdir+File.separator+fileName+"_s"+(i+1)+".tiff");
			}
			print("Saving "+fileName+"_s"+(i+1)+" under "+outputdir);
			run("Close All");
		}
	}
}
Ext.close();
setBatchMode(false);
showMessage("Done with "+k+" files and "+n+" series!");