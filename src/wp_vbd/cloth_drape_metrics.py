"""Saved-shape measures for this demo's broad, single-corner cloth drape.

Area preservation alone cannot detect a sheet folded into a narrow bundle.
These are appearance diagnostics, not universal cloth validity constraints.
"""
import numpy as np


def drape_metrics(rest, positions, faces, corner_id, *, grasp_radius=.035):
    rest=np.asarray(rest,dtype=np.float64)
    x=np.asarray(positions,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    if x.ndim!=2 or x.shape!=rest.shape or x.shape[1]!=3:
        raise ValueError('Matching rest/current cloth vertices required')
    triangles=rest[faces]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=np.linalg.norm(normal,axis=-1)
    if np.any(area<=1e-15):raise ValueError('Nondegenerate rest triangles required')
    normal/=area[:,None]
    _,axes=np.linalg.eigh((normal.T*area)@normal)
    dominant=axes[:,-1]
    # Exclude the authored grip fold and the immediate finger patch.
    panel=(abs(normal@dominant)>np.cos(np.deg2rad(20.)))
    panel&=(np.linalg.norm(triangles.mean(1)-rest[corner_id],axis=-1)>grasp_radius)
    if panel.sum()<4:raise ValueError('A broad cloth panel is required for this drape diagnostic')
    selected=faces[panel];ids=np.unique(selected)
    reference=np.linalg.svd(rest[ids]-rest[ids].mean(0),compute_uv=False)
    spectrum=np.linalg.svd(x[ids]-x[ids].mean(0),compute_uv=False)
    current=x[selected]
    normals=np.cross(current[:,1]-current[:,0],current[:,2]-current[:,0])
    weights=np.linalg.norm(normals,axis=-1)
    if np.any(weights<=1e-15):
        return dict(panel_width_ratio=0.,panel_normal_coherence=0.,severe_panel_hinges=0,
                    panel_hinges=0,panel_triangles=int(panel.sum()),degenerate=True)
    normals/=weights[:,None]
    coherence=float(np.linalg.eigvalsh((normals.T*weights)@normals/weights.sum())[-1])
    adjacent={}
    for i,face in enumerate(selected):
        for a,b in zip(face,np.roll(face,-1)):
            adjacent.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    angles=[]
    for pair in adjacent.values():
        if len(pair)==2:
            angles.append(float(np.arccos(np.clip(normals[pair[0]]@normals[pair[1]],-1.,1.))))
    return dict(panel_width_ratio=float(spectrum[1]/max(reference[1],1e-15)),
                panel_normal_coherence=coherence,
                severe_panel_hinges=sum(angle>np.pi/2 for angle in angles),
                panel_hinges=len(angles),panel_triangles=int(panel.sum()),degenerate=False)
