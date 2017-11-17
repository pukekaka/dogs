import idautils
import idc

for ea in idautils.Segments():

    segend = idc.GetSegmentAttr(ea, idc.SEGATTR_END)
    start = ea
    while start < segend:
        idc.MakeCode(start)
        start = idc.FindUnexplored(start+1, idc.SEARCH_DOWN)

idc.GenerateFile(idc.OFILE_ASM, idc.GetInputFile()+".asm", 0, idc.BADADDR, 0)

idc.Exit(0)

# Here it is. Run it with idal -c -A -S./script.py ./test.bin