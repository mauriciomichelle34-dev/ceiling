const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'..','planner.js'),'utf8');
const start=source.indexOf('function coordinateOrigin(');
const end=source.indexOf('\nfunction calibration(',start);
function fixture() {
  const origin={kind:'fixed_screw',x_px:3884.5,y_px:2963.5,image_width:5472,image_height:3648};
  const meta={width:5472,height:3648,screw_origin:{status:'detected',origin}};
  const context={originMode:'auto_screw',backendOrigin:{image_width:5472,image_height:3648},current:{meta}};
  vm.createContext(context);vm.runInContext(source.slice(start,end),context);
  return {context,meta,origin};
}
test('auto mode draws and converts using each image origin, never the old fixed pixel',()=>{
  const {context,meta}=fixture();
  assert.equal(context.coordinateOrigin().x_px,3884.5);
  const second={...meta,screw_origin:{status:'detected',origin:{...meta.screw_origin.origin,x_px:4043.5,y_px:3010.5}}};
  assert.equal(context.coordinateOrigin(second).x_px,4043.5);
  assert.equal(context.coordinateOrigin().x_px,3884.5);
});
test('explicit last-success fallback remains usable',()=>{
  const {context,meta}=fixture();meta.screw_origin.status='reused_last';
  assert.equal(context.coordinateOrigin().y_px,2963.5);
});
test('legacy metadata, invalid positions and mismatched frames cannot generate coordinates',()=>{
  for(const modify of [m=>delete m.screw_origin,m=>m.screw_origin.status='failed',
    m=>m.screw_origin.origin.x_px=NaN,m=>m.screw_origin.origin.x_px=5472,
    m=>m.screw_origin.origin.image_width=100,m=>m.width=100]) {
    const {context,meta}=fixture();modify(meta);assert.throws(()=>context.coordinateOrigin());
  }
});
