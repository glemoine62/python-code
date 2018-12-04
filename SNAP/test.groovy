import static groovy.io.FileType.FILES
def dir = new File("/home/lemoigu/.snap/auxdata/dem/SRTM 1Sec HGT")
def files = []
dir.traverse(type: FILES, maxDepth: 0) { files.add(it) }

for (f in files) {
	def test=SRTM1HgtFileInfo.create(f)
	println test.easting
	println test.northing
	println test.fileName
}