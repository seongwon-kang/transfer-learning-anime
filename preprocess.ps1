# Randomize files
# Get-ChildItem * | Rename-Item -NewName { $_.name + ".PNG" }

# Convert raw files
ls raw | Foreach{ python .\bulk_convert.py raw/$_.name cropped/$_.name/}

# Resize 
ls cropped | Foreach{python .\bulk_resize.py cropped/$_ resized/$_ }

# Extract to test
ls cropped | Foreach { $dir = $_; cd $dir; (ls | Sort CreationTime -Descending | Select-Object -First 4 Name) | Foreach {mv $_.Name ../../test/$dir/$_.Name}; cd ..}