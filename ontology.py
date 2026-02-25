from owlready2 import *

onto=get_ontology("http://example.org/house_ontology.owl")

with onto:
    class Room (Thing):
        pass

    class LivingRoom(Room):
        pass

    class Sofa(Thing):
        pass

    class Telivision(Thing):
        pass

    #properties
    class hasFurniture(ObjectProperty):
        domain=[Room]
        range=[Thing]

    class hasDevice(ObjectProperty):
        domain=[Room]
        range=[Thing]

#create instances
living_room=LivingRoom("MyLivingRoom")

sofa=Sofa("MySofa")
tv=Telivision("MyTV")

living_room.hasFurniture.append(sofa)
living_room.hasDevice.append(tv)

onto.save(file="living_room.owl")

print("Ontology created successfully")