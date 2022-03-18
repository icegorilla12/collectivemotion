import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import uniform
import matplotlib.animation as animation

class Agent:
    def __init__(self,ini_pos, ini_velocity,ini_direction,ini_angle,cell,agentId):
        self.position = ini_pos
        self.velocity = ini_velocity
        self.direction = ini_direction
        self.angle = ini_angle
        self.ID = agentId
        self.cell = cell

def findDirection(angle):
    if 0<angle<np.pi/2:
        return np.array([np.cos(angle), np.sin(angle)])
    elif np.pi/2<angle<np.pi:
        return np.array([-np.sin(angle-np.pi/2), np.cos(angle-np.pi/2)])
    elif np.pi<angle<(3/2)*np.pi:
        return np.array([-np.cos(angle-np.pi), -np.sin(angle-np.pi)])
    elif (3/2)*np.pi< angle <= 2*np.pi:
        return np.array([np.sin(angle-(np.pi*(3/2))), -np.cos(angle-np.pi*(3/2))])
    elif angle==0:
        return np.array([1,0])
    elif angle==np.pi/2:
        return np.array([0,1])
    elif angle == (3/2)*np.pi:
        return np.array([0,-1])
    elif angle == 2*np.pi:
        return np.array([1,0])

def InitialiseParticles(N,Lx,Ly, velocity = 5):
    agents = []
    for i in range(N):
        pos = np.asarray([uniform(Lx/2 - 6,Lx/2+ 6),uniform(Ly/2 -6, Ly/2+ 6)])
        angle = uniform(0,2*np.pi)
        direction = findDirection( angle)
        agents.append(Agent(pos,velocity,direction,angle,0,i+1))
    return agents

def getValue(ix,iy,leny):
    return int(ix*(leny-1) + iy)

def makeCellsGrid(agents,radius, max_occupants,Lx,Ly):
    x = np.arange(0, Lx+radius, radius)
    y = np.arange(0, Ly+radius, radius)
    lenx = len(x)
    leny = len(y)
    make_map = np.arange((lenx-1)*(leny-1))
    make_map = make_map.reshape((lenx-1),(leny-1))
    grid = np.zeros(((lenx-1),(leny-1),max_occupants))
    grid_next = np.zeros((lenx-1)*(leny-1))
    for i,a in enumerate(agents):
        index_x = a.position[0]//radius
        index_y = a.position[1]//radius
        val =  getValue(index_x,index_y,leny)
        a.cell = val
        grid[int(index_x)][int(index_y)][int(grid_next[val])] = a.ID
        grid_next[int(val)]+=1
    return grid.astype(int), make_map.astype(int)

def ifNeighbour(agent1, agent2,Rr, Ral,Rattr):
    if(np.sqrt(np.sum(np.square(agent1.position - agent2.position)))<=Rr):
        return 1
    elif(Rr<np.sqrt(np.sum(np.square(agent1.position - agent2.position)))<=Ral):
        return 2
    elif(Ral<np.sqrt(np.sum(np.square(agent1.position - agent2.position)))<=Rattr):
        return 3
    

def findNearest(agents, grid, map, max_occupants,Rr, Ral, Rattr):
    neigh_listR = np.zeros((len(agents)+1,max_occupants))
    neigh_listAl = np.zeros((len(agents)+1,max_occupants))
    neigh_listAttr = np.zeros((len(agents)+1,max_occupants))
    for a in agents:
        flag1 = 0
        flag2 = 0
        flag3 = 0
        cell = a.cell
        ix = np.where(map==cell)[0][0]
        iy =  np.where(map==cell)[1][0]
        for i in range(max_occupants):
            if(grid[ix][iy][i]!=0 and grid[ix][iy][i]!=a.ID):
                if(ifNeighbour(a,agents[grid[ix][iy][i]-1],Rr,Ral,Rattr)==1):
                    neigh_listR[a.ID][flag1] = grid[ix][iy][i]
                    flag1+=1
                elif(ifNeighbour(a,agents[grid[ix][iy][i]-1],Rr,Ral,Rattr)==2):
                    neigh_listAl[a.ID][flag2] = grid[ix][iy][i]
                    flag2+=1
                elif(ifNeighbour(a,agents[grid[ix][iy][i]-1],Rr,Ral,Rattr)==3):
                    neigh_listAttr[a.ID][flag3] = grid[ix][iy][i]
                    flag3+=1
            else:
                    break
        for i in range(max_occupants):
            if(ix-1>=0):
                if(grid[ix-1][iy][i]!=0): 
                    if(ifNeighbour(a,agents[grid[ix-1][iy][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix-1][iy][i]
                        flag1+=1
                    elif(ifNeighbour(a,agents[grid[ix-1][iy][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix-1][iy][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix-1][iy][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix-1][iy][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(ix+1<grid.shape[0]):
                if(grid[ix+1][iy][i]!=0): 
                    if(ifNeighbour(a,agents[grid[ix+1][iy][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix+1][iy][i]
                        flag1+=1
                    elif(ifNeighbour(a,agents[grid[ix+1][iy][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix+1][iy][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix+1][iy][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix+1][iy][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy+1<grid.shape[1]):
                if(grid[ix][iy+1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix][iy+1][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix][iy+1][i]
                        flag1+=1
                    elif(ifNeighbour(a,agents[grid[ix][iy+1][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix][iy+1][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix][iy+1][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix][iy+1][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy-1>=0):
                if(grid[ix][iy-1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix][iy-1][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix][iy-1][i]
                        flag1+=1
                    if(ifNeighbour(a,agents[grid[ix][iy-1][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix][iy-1][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix][iy-1][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix][iy-1][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy-1>=0 and ix-1>=0):
                if(grid[ix-1][iy-1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix-1][iy-1][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix-1][iy-1][i]
                        flag1+=1
                    if(ifNeighbour(a,agents[grid[ix-1][iy-1][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix-1][iy-1][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix-1][iy-1][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix-1][iy-1][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy+1<grid.shape[1] and ix-1>=0):
                if(grid[ix-1][iy+1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix-1][iy+1][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix-1][iy+1][i]
                        flag1+=1
                    elif(ifNeighbour(a,agents[grid[ix-1][iy+1][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix-1][iy+1][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix-1][iy+1][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix-1][iy+1][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy+1<grid.shape[1] and ix+1<grid.shape[0]):
                if(grid[ix+1][iy+1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix+1][iy+1][i]-1],Rr,Ral,Rattr)==1):
                        neigh_listR[a.ID][flag1] = grid[ix+1][iy+1][i]
                        flag1+=1
                    elif(ifNeighbour(a,agents[grid[ix+1][iy+1][i]-1],Rr,Ral,Rattr)==2):
                        neigh_listAl[a.ID][flag2] = grid[ix+1][iy+1][i]
                        flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix+1][iy+1][i]-1],Rr,Ral,Rattr)==3):
                        neigh_listAttr[a.ID][flag3] = grid[ix+1][iy+1][i]
                        flag3+=1
                else:
                    break
            else:
                    break
        for i in range(max_occupants):
            if(iy-1>0 and ix+1<grid.shape[0]):
                if(grid[ix+1][iy-1][i]!=0):
                    if(ifNeighbour(a,agents[grid[ix+1][iy-1][i]-1],Rr,Ral,Rattr)==1):
                            neigh_listR[a.ID][flag1] = grid[ix+1][iy-1][i]
                            flag1+=1
                    if(ifNeighbour(a,agents[grid[ix+1][iy-1][i]-1],Rr,Ral,Rattr)==2):
                            neigh_listAl[a.ID][flag2] = grid[ix+1][iy-1][i]
                            flag2+=1
                    elif(ifNeighbour(a,agents[grid[ix+1][iy-1][i]-1],Rr,Ral,Rattr)==3):
                            neigh_listAttr[a.ID][flag3] = grid[ix+1][iy-1][i]
                            flag3+=1
                    else:
                        break
            else:
                    break
    return neigh_listR, neigh_listAl, neigh_listAttr

def getNeighList(agents,max_occupants,Lx,Ly,Rr,Ral,Rattr):
    grid, map = makeCellsGrid(agents,Rattr, max_occupants,Lx,Ly)
    neigh_listR, neigh_listAl, neigh_listAttr = findNearest(agents, grid, map,max_occupants,Rr,Ral,Rattr)
    return neigh_listR.astype(int), neigh_listAl.astype(int), neigh_listAttr.astype(int)

def getAngle(direction):
    if(direction[0]>0 and direction[1]>0):
        return np.arctan(direction[1]/direction[0])
    elif(direction[0]<0 and direction[1]>0):
        return np.pi/2 + np.arctan(np.abs(direction[0])/direction[1])
    elif(direction[0]<0 and direction[1]<0):
        return np.pi + np.arctan(np.abs(direction[1])/np.abs(direction[0]))
    elif(direction[0]>0 and direction[1]<0):
        return np.pi*(3/2) + np.arctan(np.abs(direction[0])/np.abs(direction[1]))
    elif(direction[0]==0 and direction[1]<0):
        return np.pi*(3/2)
    elif(direction[0]==0 and direction[1]>0):
        return np.pi/2
    elif(direction[0]>0 and direction[1]==0):
        return 0
    elif(direction[0]<0 and direction[1]==0):
        return np.pi
    else:
        return 0


def update(agents, max_occupants , iterations, Lx, Ly, Rr, Ral, Rattr, MI = 0.4*0.5**2, torque_parameter=0.01, alpha_r=0.2, alpha_align=0.7, alpha_attr=0.1,  time_step=0.1):
    neighListR, neighListalign, neighListattr = getNeighList(agents,max_occupants,Lx,Ly,Rr,Ral,Rattr)
    angles = []
    directions = []
    positions = []
    velocities = []
    for i in range(2):
        temp = []
        for a in agents:
            temp.append(a.angle)
        angles.append(temp)
    for i in range(1):
        temp1 = []
        temp2 = []
        temp3 = []
        for a in agents:
            temp1.append(a.direction)
            temp2.append(a.position)
            temp3.append(a.velocity)
        directions.append(temp1)
        positions.append(temp2)
    for iter in range(iterations):
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        for i,a in enumerate(agents):
            w_r= np.asarray([0 ,0]).astype(np.float32)
            w_attr = np.asarray([0 ,0]).astype(np.float32)
            w_align = np.asarray([0 ,0]).astype(np.float32)
            angle_ini = a.angle
            for j in range(max_occupants):
                if(neighListR[i+1][j]!=0):
                    w_r+=a.position - agents[neighListR[i+1][j]-1].position
                else:
                    break
            count = 0
            for j in range(max_occupants):
                if(neighListalign[i+1][j]!=0):
                    count+=1
                    w_align+=agents[neighListalign[i+1][j]-1].direction
                    # print(agents[neighListalign[i+1][j]-1].direction)
                else:
                    break
            # print(w_align)
            if(count==0):
                w_align = np.asarray([0 ,0]).astype(np.float32)
            else:
                w_align = w_align/count
            for j in range(max_occupants):
                if(neighListattr[i+1][j]!=0):
                    w_attr+=(agents[neighListattr[i+1][j]-1].position - a.position)
                else:
                    break
            if(np.sqrt(np.sum(np.square(w_r)))!=0):
                w_r = w_r/np.sqrt(np.sum(np.square(w_r)))
            if(np.sqrt(np.sum(np.square(w_attr)))!=0):
                w_attr = w_attr/np.sqrt(np.sum(np.square(w_attr)))
            if(np.sqrt(np.sum(np.square(w_align)))!=0):
                w_align = w_align/np.sqrt(np.sum(np.square(w_align)))

            w = alpha_r*w_r + alpha_align*w_align + alpha_attr*w_attr

            if(np.sqrt(np.sum(np.square(w)))!=0):
                w = w/np.sqrt(np.sum(np.square(w)))
            else:
                w = a.direction
            
            angle_final = getAngle(w)
            
            if(angle_final > angle_ini):
                torque = torque_parameter*(min(np.abs(angle_final - angle_ini),np.abs(2*np.pi - (angle_final -angle_ini))))
            elif(angle_final < angle_ini):
                torque = -torque_parameter*(min(np.abs(angle_final - angle_ini),np.abs(2*np.pi - (angle_final -angle_ini))))
            else:
                torque = 0
            # torque = 0
            # print(angle_final)
            
            after_angle = 2*angles[iter+1][a.ID-1] - angles[iter][a.ID-1] + (time_step**2)*(torque)/MI
            if(angle_final >  angle_ini):
                if(after_angle > angle_final):
                    after_angle = angle_final
            elif(angle_final <= angle_ini):
                if(after_angle < angle_final):
                    after_angle = angle_final
            if(after_angle>2*np.pi):
                after_angle = after_angle%(2*np.pi)
            if(after_angle<0):
                after_angle = 0
            a.angle = after_angle
            a.direction = findDirection(a.angle)
            temp1.append(a.angle)
            temp2.append(a.direction)
            a.position = a.position + a.velocity*a.direction*time_step
            temp3.append(a.position)
            temp4.append(a.velocity)
        angles.append(temp1)
        directions.append(temp2)
        positions.append(temp3)
        velocities.append(temp4)
        neighListR, neighListalign, neighListattr = getNeighList(agents,max_occupants,Lx,Ly,Rr,Ral,Rattr)

    return angles,directions,positions,velocities, agents

def make_animation(agentsPositions,agentsDirections,output_file, Lx, Ly, framesPerSecond=25):

# Settings
  video_file = output_file
  clear_frames = False    # Should it clear the figure between each frame?
  fps = framesPerSecond
  # Output video writer
  FFMpegWriter = animation.writers['ffmpeg']
  metadata = dict(title='Collective Motion', artist='Matplotlib', comment='Move')
  writer = FFMpegWriter(fps=fps, metadata=metadata)
  fig, ax = plt.subplots()
  plt.tick_params(
      axis='x',         
      which='both',     
      bottom=False,     
      top=False,         
      labelbottom=False)
  plt.title('Collective Motion')
  with writer.saving(fig, video_file, 100):
      for i in range(len(agentsPositions)):
            ax.set_xlim(Lx)
            ax.set_ylim(Ly)
            ax.scatter(agentsPositions[i][:,0],agentsPositions[i][:,1],color='blue', label = 'Agents', edgecolors = 'black', s=10, zorder=1, alpha = 0.8)
            ax.quiver(agentsPositions[i][:,0],agentsPositions[i][:,1],agentsDirections[i][:,0] , agentsDirections[i][:,1],  color='black', units='inches' , angles='xy', scale=5,width=0.015,headlength=3,headwidth=2,alpha=0.8)
            writer.grab_frame()
            ax.clear()
        
  plt.clf()











    





    
    

    









