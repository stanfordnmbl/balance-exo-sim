
def create_contact_sphere_force_table(self, model, statesTrajectory):

    model.initSystem()
    externalForcesTable = osim.TimeSeriesTableVec3()
    numStates = statesTrajectory.getSize() 

    forceNames = ['forceset/contactHeel_l',
                  'forceset/contactLateralRearfoot_l',
                  'forceset/contactLateralMidfoot_l',
                  'forceset/contactLateralToe_l',
                  'forceset/contactMedialToe_l',
                  'forceset/contactMedialMidfoot_l',
                  'forceset/contactHeel_r',
                  'forceset/contactLateralRearfoot_r',
                  'forceset/contactLateralMidfoot_r',
                  'forceset/contactLateralToe_r',
                  'forceset/contactMedialToe_r',
                  'forceset/contactMedialMidfoot_r']

    forceLabels = ['heel_l', 'lat_rear_l', 'lat_mid_l', 
                   'lat_toe_l', 'med_toe_l', 'med_mid_l',
                   'heel_r', 'lat_rear_r', 'lat_mid_r', 
                   'lat_toe_r', 'med_toe_r', 'med_mid_r']

    sphereNames = ['contactgeometryset/heel_l',
                   'contactgeometryset/lateralRearfoot_l',
                   'contactgeometryset/lateralMidfoot_l',
                   'contactgeometryset/lateralToe_l',
                   'contactgeometryset/medialToe_l',
                   'contactgeometryset/medialMidfoot_l',
                   'contactgeometryset/heel_r',
                   'contactgeometryset/lateralRearfoot_r',
                   'contactgeometryset/lateralMidfoot_r',
                   'contactgeometryset/lateralToe_r',
                   'contactgeometryset/medialToe_r',
                   'contactgeometryset/medialMidfoot_r']


    for istate in range(numStates):

        state = statesTrajectory.get(istate)
        model.realizeVelocity(state)

        row = osim.RowVectorVec3(3*len(forceNames))
        labels = osim.StdVectorString()
        zipped = zip(forceNames, forceLabels, sphereNames)
        for i, forceName, forceLabel, sphereName in enumerate(zipped):

            force = osim.Vec3(0)
            torque = osim.Vec3(0)

            force = osim.Force.safeDownCast(model.getComponent(forceName))
            forceValues = force.getRecordValues(state)

            force[0] = forceValues[0]
            force[1] = forceValues[1]
            force[2] = forceValues[2]
            torque[0] = forceValues[3]
            torque[1] = forceValues[4]
            torque[2] = forceValues[5]

            sphere = osim.ContactSphere.safeDownCast(model.getComponent())
            frame = sphere.getFrame()
            position = sphere.getPositionInGround(state)

            row.set(3*i, force)
            row.set(3*i + 1, torque)
            row.set(3*i + 2, position)

            for suffix in ['_force_v', '_torque_', '_force_p']:
                labels.append(f'{forceLabel}{suffix}')

        externalForcesTable.appendRow(state.getTime(), row)

    externalForcesTable.setColumnLabels(labels)

    return externalForcesTable.flatten({"x", "y", "z"})

    
def create_external_loads_table_for_gait(model,
        statesTrajectory, forcePathsRightFoot, forcePathsLeftFoot):
    model.initSystem()
    externalForcesTable = osim.TimeSeriesTableVec3()
    numStates = statesTrajectory.getSize() 

    for istate in range(numStates):

        state = statesTrajectory.get(istate)
        model.realizeVelocity(state)

        forcesRight = osim.Vec3(0)
        torquesRight = osim.Vec3(0)
        copRight = osim.Vec3(0)

        # Loop through all Forces of the right side.
        for smoothForce in forcePathsRightFoot:
            force = osim.Force.safeDownCast(model.getComponent(smoothForce))
            forceValues = force.getRecordValues(state)
            forcesRight[0] = forcesRight[0] + forceValues[0]
            forcesRight[1] = forcesRight[1] + forceValues[1]
            forcesRight[2] = forcesRight[2] + forceValues[2]
            torquesRight[0] = torquesRight[0] + forceValues[3]
            torquesRight[1] = torquesRight[1] + forceValues[4]
            torquesRight[2] = torquesRight[2] + forceValues[5]


        copRight[0] = torquesRight[2] / forcesRight[1]
        copRight[1] = 0
        copRight[2] = -torquesRight[0] / forcesRight[1]

        torquesRight[0] = 0
        torquesRight[1] = torquesRight[1] - copRight[2]*forcesRight[0] + copRight[0]*forcesRight[2]  
        torquesRight[2] = 0

        forcesLeft = osim.Vec3(0)
        torquesLeft = osim.Vec3(0)
        copLeft = osim.Vec3(0)


        # Loop through all Forces of the left side.
        for smoothForce in forcePathsLeftFoot:
            force = osim.Force.safeDownCast(model.getComponent(smoothForce))
            forceValues = force.getRecordValues(state)
            forcesLeft[0] = forcesLeft[0] + forceValues[0]
            forcesLeft[1] = forcesLeft[1] + forceValues[1]
            forcesLeft[2] = forcesLeft[2] + forceValues[2]
            torquesLeft[0] = torquesLeft[0] + forceValues[3]
            torquesLeft[1] = torquesLeft[1] + forceValues[4]
            torquesLeft[2] = torquesLeft[2] + forceValues[5]


        copLeft[0] = torquesLeft[2] / forcesLeft[1]
        copLeft[1] = 0
        copLeft[2] = -torquesLeft[0] / forcesLeft[1]

        torquesLeft[0] = 0
        torquesLeft[1] = torquesLeft[1] - copLeft[2]*forcesLeft[0] + copLeft[0]*forcesLeft[2]  
        torquesLeft[2] = 0

        # Append row to table.
        row = osim.RowVectorVec3(6)
        row.set(0, forcesRight)
        row.set(1, copRight)
        row.set(2, forcesLeft)
        row.set(3, copLeft)
        row.set(4, torquesRight)
        row.set(5, torquesLeft)
        externalForcesTable.appendRow(state.getTime(), row);
    
    # Create table.
    labels = osim.StdVectorString()
    labels.append('ground_force_r_v')
    labels.append('ground_force_r_p')
    labels.append('ground_force_l_v')
    labels.append('ground_force_l_p')
    labels.append('ground_torque_r_')
    labels.append('ground_torque_l_')

    externalForcesTable.setColumnLabels(labels)
    return externalForcesTable.flatten({"x", "y", "z"})

