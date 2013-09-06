import numpy as np
import geom
import utils
import vtk
from vtk.util import numpy_support

n = 200
d = 3
L = 1.0

R = 0.02
l = 0.08
l_half = l / 2.0

r0 = np.zeros([d])
Rd = 0.5

r = np.zeros([n, d])
u = np.zeros_like(r)

i = 0
for i in range(n):
    while True:
        r[i] = np.random.uniform(-L/2.0, L/2.0, size=d)
        u[i] = utils.sphere_pick(d)
        valid = True
        if geom.cap_insphere_intersect(r[i] - l_half*u[i], r[i] + l_half*u[i], R, r0, Rd):
            valid = False
        for i2 in range(i):
            if geom.caps_intersect(r[i] - l_half*u[i], r[i] + l_half*u[i], R, r[i2] - l_half*u[i2], r[i2] + l_half*u[i2], R):
                valid = False
        if valid: break

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(600, 600)
renWin.AddRenderer(ren)
# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()

# System bounds
sys = vtk.vtkCubeSource()
sys.SetXLength(L)
sys.SetYLength(L)
sys.SetZLength(L)
sysMapper = vtk.vtkPolyDataMapper()
sysMapper.SetInputConnection(sys.GetOutputPort())
sysActor = vtk.vtkActor()
sysActor.GetProperty().SetOpacity(0.2)
sysActor.SetMapper(sysMapper)
ren.AddActor(sysActor)

envMapper = vtk.vtkPolyDataMapper()
envActor = vtk.vtkActor()
env = vtk.vtkSphereSource()
env.SetThetaResolution(30)
env.SetPhiResolution(30)
env.SetRadius(Rd)
envMapper.SetInputConnection(env.GetOutputPort())
envActor.SetMapper(envMapper)
envActor.GetProperty().SetColor(1, 0, 0)
envActor.GetProperty().SetOpacity(0.2)
ren.AddActor(envActor)

particleCPoints = vtk.vtkPoints()
particleCPolys = vtk.vtkPolyData()
particleCPolys.SetPoints(particleCPoints)
particlesC = vtk.vtkGlyph3D()

lineSource = vtk.vtkLineSource()
lineSource.SetPoint1(-l_half, 0.0, 0.0)
lineSource.SetPoint2(l_half, 0.0, 0.0)
particleCSource = vtk.vtkTubeFilter()
particleCSource.SetInputConnection(lineSource.GetOutputPort())
particleCSource.SetRadius(R)
particleCSource.SetNumberOfSides(20)

particlesC.SetSourceConnection(particleCSource.GetOutputPort())
particlesC.SetInputData(particleCPolys)
particlesCMapper = vtk.vtkPolyDataMapper()
particlesCMapper.SetInputConnection(particlesC.GetOutputPort())
particlesCActor = vtk.vtkActor()
particlesCActor.SetMapper(particlesCMapper)
# particlesCActor.GetProperty().SetColor(0, 1, 0)
particleCPoints.SetData(numpy_support.numpy_to_vtk(r))
particleCPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(u))
ren.AddActor(particlesCActor)

particleESource = vtk.vtkSphereSource()
particleESource.SetRadius(R)
particleESource.SetThetaResolution(20)
particleESource.SetPhiResolution(20)

particleE1Points = vtk.vtkPoints()
particleE1Polys = vtk.vtkPolyData()
particleE1Polys.SetPoints(particleE1Points)
particlesE1 = vtk.vtkGlyph3D()
particlesE1.SetSourceConnection(particleESource.GetOutputPort())
particlesE1.SetInputData(particleE1Polys)
particlesE1Mapper = vtk.vtkPolyDataMapper()
particlesE1Mapper.SetInputConnection(particlesE1.GetOutputPort())
particlesE1Actor = vtk.vtkActor()
particlesE1Actor.SetMapper(particlesE1Mapper)
# particlesE1Actor.GetProperty().SetColor(1, 0, 0)
re1 = r + u * l_half
particleE1Points.SetData(numpy_support.numpy_to_vtk(re1))
ren.AddActor(particlesE1Actor)

particleE2Points = vtk.vtkPoints()
particleE2Polys = vtk.vtkPolyData()
particleE2Polys.SetPoints(particleE2Points)
particlesE2 = vtk.vtkGlyph3D()
# particleE2Source = vtk.vtkSphereSource()
# particleE2Source.SetRadius(R)
particlesE2.SetSourceConnection(particleESource.GetOutputPort())
particlesE2.SetInputData(particleE2Polys)
particlesE2Mapper = vtk.vtkPolyDataMapper()
particlesE2Mapper.SetInputConnection(particlesE2.GetOutputPort())
particlesE2Actor = vtk.vtkActor()
particlesE2Actor.SetMapper(particlesE2Mapper)
# particlesE2Actor.GetProperty().SetColor(0, 0, 1)
re2 = r - u * l_half
particleE2Points.SetData(numpy_support.numpy_to_vtk(re2))
ren.AddActor(particlesE2Actor)

renWin.Render()
iren.Start()