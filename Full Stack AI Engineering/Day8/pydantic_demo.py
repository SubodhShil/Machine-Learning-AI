from pydantic import BaseModel

class Student(BaseModel):
    name: str

new_student = {'name': 'Subodh'}