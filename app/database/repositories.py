from sqlalchemy.orm import Session
from app.database.models import Document, Conversation, Message
import uuid

def create_document(db: Session, user_id: str, filename: str, file_type: str) -> Document:
    doc = Document(id=str(uuid.uuid4()), user_id=user_id, filename=filename, file_type=file_type)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

def get_documents(db: Session, user_id: str) -> list[Document]:
    return db.query(Document).filter(Document.user_id == user_id).all()

def get_document(db: Session, document_id: str) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()

def delete_document(db: Session, document_id: str) -> bool:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True

def get_or_create_conversation(db: Session, conversation_id: str | None, user_id: str) -> Conversation:
    if conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            return conv
    conv = Conversation(id=conversation_id or str(uuid.uuid4()), user_id=user_id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def add_message(db: Session, conversation_id: str, role: str, content: str) -> Message:
    msg = Message(id=str(uuid.uuid4()), conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_recent_messages(db: Session, conversation_id: str, limit: int = 10) -> list[Message]:
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(messages))
