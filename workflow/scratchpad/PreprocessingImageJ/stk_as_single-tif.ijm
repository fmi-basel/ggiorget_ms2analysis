/*
 * Resaving Macro for stk to tif
 * By Jana Tuennermann
 * Given one or several *.stk files (either single stage or multi stage format), resaves as single tif file into a given output folder (one file per stage position)
 */

#@ File[] (label="Select some files to be resaved", style="files") listfiles
#@ File (label="Select output directory", style="directory") outputdir

setBatchMode(true);

for (i=0; i<listfiles.length; i++) {
	open(listfiles[i]);
	if (nSlices>1){
		fullName = getTitle();
		newName = replace(fullName, ".stk", "_s");
		rename(newName);
		run("Image Sequence... ", "format=TIFF save=["+outputdir+"] start=1 digits=1"); 
		print("Resaved " + fullName);
    	close();
	}else {
		fullName = getTitle();
		newName = replace(fullName, ".stk", "_s1");
		rename(newName);
		fullName = getTitle();
		saveAs("tiff", outputdir+File.separator+fullName+".tif");
		print("Resaved " + fullName);
    	close();
	}
	
}
